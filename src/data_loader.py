import os.path
from collections import defaultdict
import numpy as np
import torch, random
import networkx as nx
from scipy.sparse import csr_matrix
from tqdm import tqdm
from utils import *


class DataLoader:
    def __init__(self, args, data_dir, dataset):
        self.args = args
        self.name = data_dir.split('/')[-2]

        self.kg = KG(args, data_dir)
        self.build_graph(self.kg.background)
        self.name = dataset

    def get_batch(self, batch_idx, mode='train'):
        if mode == 'train':
            return self.kg.data[batch_idx], self.kg.entity_num
        else:
            query, answer = np.array(self.kg.query), np.array(self.kg.answer, dtype=object)
            subs = query[batch_idx, 0]
            rels = query[batch_idx, 1]
            objs = np.zeros((len(batch_idx), self.kg.entity_num))
            for i in range(len(batch_idx)):
                objs[i][answer[batch_idx[i]]] = 1
            return subs, rels, objs

    def get_case_graph(self, rels, id=None):
        return self.kg.get_case_graph(rels, id)

    def get_neighbors(self, nodes, subs, rels, objs=None, mask_relations=None, training=False):
        indices_ = torch.cat([torch.from_numpy(nodes[:, 1]).cuda().long().unsqueeze(1), torch.from_numpy(nodes[:, 0]).cuda().long().unsqueeze(1)], dim=1).t()
        node_1hot = torch.sparse_coo_tensor(indices_, torch.ones(len(nodes)).cuda(),
                                            torch.Size([self.kg.entity_num, nodes.shape[0]]))

        edge_1hot = torch.sparse.mm(self.M_sub, node_1hot)  # edge_idx x batch_idx
        edges = edge_1hot.indices()


        selected_edges = torch.index_select(self.graph, 0, edges[0])
        sampled_edges = torch.cat([edges[1].unsqueeze(1), selected_edges], dim=1).long()
        if training:
            mask = sampled_edges[:, 2] == self.kg.relation_num  # self-loop edges
            if self.args.use_augment:
                for rel in mask_relations:  # del mask_relations
                    mask = mask | (sampled_edges[:, 2] == rel)
                for i in range(len(subs)):  # recover query relations
                    rels_mask = (sampled_edges[:, 2] == rels[i])
                    mask = mask & ~rels_mask
            sampled_edges = sampled_edges[~mask]

        # Now, remove edges corresponding to subs, rels, and objs for each sample in the batch
        if training:# and not finetune:
            for i in range(len(subs)):
                sample_mask = (sampled_edges[:, 0] == i)
                if objs is not None:
                    subs_mask = (sampled_edges[:, 1] == subs[i])
                    rels_mask = (sampled_edges[:, 2] == rels[i])
                    objs_mask = (sampled_edges[:, 3] == objs[i])

                    # Combine all the masks to filter out the unwanted edges
                    final_mask = sample_mask & subs_mask & rels_mask & objs_mask

                    subs_mask = (sampled_edges[:, 3] == subs[i])
                    rel_inv = self.kg.relation_num//2 + rels[i] if rels[i] < self.kg.relation_num//2 else rels[i] - self.kg.relation_num//2
                    rels_mask = (sampled_edges[:, 2] == rel_inv)
                    objs_mask = (sampled_edges[:, 1] == objs[i])

                    # Combine all the masks to filter out the unwanted edges
                    final_mask = final_mask | (sample_mask & subs_mask & rels_mask & objs_mask)
                else:
                    final_mask = torch.zeros([len(sample_mask)])
                final_mask = final_mask.bool()
                sampled_edges = sampled_edges[~final_mask]


        nodes_torch = torch.LongTensor(nodes).cuda().long()
        self_loop_edges = torch.cat([nodes_torch[:, 0].unsqueeze(1), nodes_torch[:, 1].unsqueeze(1),
                                        self.kg.relation_num * torch.ones((len(nodes), 1)).cuda().long(),
                                        nodes_torch[:, 1].unsqueeze(1)], 1)
        sampled_edges = torch.cat([sampled_edges, self_loop_edges], 0)

        head_nodes, head_index = torch.unique(sampled_edges[:, [0, 1]], dim=0, sorted=False, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:, [0, 3]], dim=0, sorted=False, return_inverse=True)

        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)

        mask = sampled_edges[:, 2] == self.kg.relation_num  # self-loop edges
        _, old_idx = head_index[mask].sort()
        old_nodes_new_idx = tail_index[mask][old_idx]

        return tail_nodes, sampled_edges, old_nodes_new_idx

    # numpy version
    # def build_graph(self, triples):
    #     self.graph = np.array(triples)
    #     self.fact_num = len(self.graph)
    #     self.M_sub = csr_matrix((np.ones((self.fact_num,)), (np.arange(self.fact_num), self.graph[:, 0])),
    #                             shape=(self.fact_num, self.kg.entity_num))
    #     self.M_obj = csr_matrix((np.ones((self.fact_num,)), (np.arange(self.fact_num), self.graph[:, 2])),
    #                             shape=(self.fact_num, self.kg.entity_num))

    # torch version
    def build_graph(self, triples):
        self.graph = torch.LongTensor(triples).cuda()
        self.fact_num = self.graph.size(0)
        indices_A = torch.cat([torch.arange(self.fact_num).long().unsqueeze(1).cuda(), self.graph[:, 0].unsqueeze(1)], dim=1).t().cuda()
        values_A = torch.ones((self.fact_num,)).cuda()
        size_A = torch.Size([self.fact_num, self.kg.entity_num])
        self.M_sub = torch.sparse_coo_tensor(indices_A, values_A, size_A).cuda()
        indices_B = torch.cat([torch.arange(self.fact_num).long().unsqueeze(1).cuda(), self.graph[:, 2].unsqueeze(1)], dim=1).t().cuda()
        self.M_obj = torch.sparse_coo_tensor(indices_B, values_A, size_A).cuda()


class KG:
    def __init__(self, args, path):
        self.args = args
        # 1. read triples
        if args.finetune and 'train' in path:
            self.data = self.load_triple(path+'/background.txt')
        else:
            self.data = self.load_triple(path+'/facts.txt')
        v_path = path.replace('train', 'valid').replace('test', 'valid')
        valid_background = self.load_triple(v_path+'/background.txt')
        valid_data = self.load_triple(v_path+'/facts.txt')
        self.answer_distance = get_distance(valid_data, valid_background)
        self.background = self.load_triple(path+'/background.txt')
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size

        # 2. read dicts
        self.entity2id = self.load_dict(path+'/entity2id.txt')
        self.relation2id = self.load_dict(path+'/relation2id.txt')
        n_rel = len(self.relation2id)
        self.relation_num = n_rel

        # add inverse relations
        self.relation2id.update({k + '_inv': v + n_rel for k, v in self.relation2id.items()})
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.id2relation = {v: k for k, v in self.relation2id.items()}
        self.entity_num = len(self.entity2id)
        self.relation_num = len(self.relation2id)
        self.h2r = self.get_h2r(self.background)

        # 3. read cases
        self.cases = defaultdict()
        for rel in tqdm(os.listdir(os.path.join(path, 'cases'))):
            rel_id = self.relation2id[rel]
            self.cases[rel_id] = defaultdict()
            self.cases[rel_id+self.relation_num//2] = defaultdict()
            case_files = os.listdir(os.path.join(path, 'cases', rel))
            num_cases = min(len(case_files)//3, int(self.args.shot))
            for case_id in range(num_cases):
                # read label
                label = self.load_case_labels(os.path.join(path, 'cases', rel) + '/' + str(case_id) + '.labels')
                triples = self.load_case_triples(os.path.join(path, 'cases', rel) + '/' + str(case_id) + '.triples')
                seen_triples = triples + [(t, r + self.relation_num//2, h) for h, r, t in triples]
                edges_index = np.array([[h, t] for h, r, t in seen_triples]).transpose()
                edges_type_ = [r for h, r, t in seen_triples]
                edges_type = np.array(edges_type_)
                edges_type_inv = np.array(edges_type_)

                query_relation = rel_id
                labels = np.array(list(label.values()))
                n_nodes = len(labels)

                self.cases[rel_id][case_id] = edges_index, edges_type, query_relation, labels, n_nodes
                self.cases[rel_id+self.relation_num//2][case_id] = edges_index, edges_type_inv, query_relation+self.relation_num//2, labels, n_nodes

        # 4. add inverse triples
        if 'FB15K-237-' not in path and 'singer' not in path and 'NELL23K' not in path:  # these datasets only have one direction queries
            self.data += [(t, r + self.relation_num//2, h) for h, r, t in self.data]

        self.background += [(t, r + self.relation_num//2, h) for h, r, t in self.background]
        self.edge_index = torch.LongTensor([[h, t] for h, r, t in self.background]).t()
        self.edge_type = torch.LongTensor([r for h, r, t in self.background])
        self.num_edges = len(self.edge_type)

        # 5. build filter
        filter = self.load_triple(path + '/filter.txt')
        filter += [(t, r + self.relation_num // 2, h) for h, r, t in filter]
        self.filter = self.get_filter(self.data, filter)

        # 6. post process
        # random.shuffle(self.data)
        self.data = np.array(list(set(self.data)))
        self.background = np.array(list(set(self.background)))
        self.query, self.answer = self.load_query(self.data)

    def load_filter(self, path):

        with open(path, 'r', encoding='utf-8') as f:
            return [int(line.split()[0]) for line in f.readlines()]

    def get_h2r(self, triples):
        h2r = defaultdict(lambda: set())
        for h, r, t in triples:
            h2r[h].add(r)
            h2r[t].add(r + self.relation_num//2)
        return h2r

    def load_query(self, triples):
        try:
            triples = triples.tolist()
        except:
            pass
        triples.sort(key=lambda x: (x[0], x[1]))
        trip_hr = defaultdict(lambda: list())

        for trip in triples:
            h, r, t = trip
            trip_hr[(h, r)].append(t)

        queries = []
        answers = []
        for idx, key in enumerate(trip_hr):
            queries.append(key)
            answers.append(np.array(trip_hr[key]))
        return queries, answers

    @staticmethod
    def load_triple(path):
        triples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                triples.append((int(h), int(r), int(t)))
        return triples

    @staticmethod
    def load_dict(path):
        key2val = dict()
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    key, val = line.strip().split('\t')
                except:
                    key, val = 'default', 0
                    print(line)
                key2val[key] = int(val)
        return key2val

    def get_filter(self, data, filter_data):
        filter = defaultdict(list)
        for h, r, t in filter_data+data:
            filter[(h, r)].append(t)
        for key, val in filter.items():
            filter[key] = list(set(val))
        return filter

    def load_case_triples(self, path):
        triples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                l = line.strip().split('\t')
                if len(l) < 3:
                    continue
                h, r, t = l
                triples.append((int(h), int(r), int(t)))
        return triples

    def load_case_labels(self, path):
        ent2label = dict()
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                ent, label1, label2 = line.strip().split('\t')
                ent2label[int(ent)] = [int(label1), int(label2)]
        return ent2label

    def case_select(self, rel, id=None):
        if id is None:
            try:
                return random.choice(list(self.cases[rel].values()))
            except:
                return np.array([[0], [1]]), np.array([rel]), rel, np.array([[0, 1], [1, 0]]), 2
        else:
            try:
                return self.cases[rel][id % len(self.cases[rel])]
            except:
                print('==========================================', rel)
                return np.array([[0], [1]]), np.array([rel]), rel, np.array([[0, 1], [1, 0]]), 2

    def get_case_graph(self, rels, random_order):
        try:
            rels = rels.cpu().tolist()
        except:
            pass
        # edge_index, edge_type, h_positions, t_positions, query_relations, num_ent
        batch_edge_index, batch_edge_type, batch_h_positions, batch_t_positions, batch_query_relations, batch_edge_query_relations = [], [], [], [], [], []
        batch_labels = []
        num_ent = 0
        for i, rel in enumerate(rels):
            if not random_order:
                id_ = i
            else:
                id_ = None
            edges_index, edges_type, query_relation, labels, num_nodes = self.case_select(rel, id=id_)

            relation_num = self.relation_num
            batch_edge_index.append(torch.from_numpy(edges_index).cuda() + num_ent)
            batch_edge_type.append(torch.from_numpy(edges_type + i * (relation_num)).cuda())
            batch_edge_query_relations.append(
                torch.ones(edges_type.shape).cuda() * query_relation + i * (self.relation_num))
            if rel < self.relation_num // 2:
                batch_h_positions.append(num_ent)
                batch_t_positions.append(num_ent + 1)
            else:
                batch_h_positions.append(num_ent + 1)
                batch_t_positions.append(num_ent)
            batch_query_relations.append(query_relation)
            batch_labels.append(torch.from_numpy(labels).cuda())
            num_ent += num_nodes
        batch_edge_index = torch.cat(batch_edge_index, -1).long()
        batch_edge_type = torch.cat(batch_edge_type, -1).long()
        batch_h_positions = torch.LongTensor(batch_h_positions).cuda().long()
        batch_t_positions = torch.LongTensor(batch_t_positions).cuda().long()
        batch_query_relations = torch.LongTensor(batch_query_relations).cuda().long()
        batch_labels = torch.cat(batch_labels, 0).long()
        batch_edge_query_relations = torch.cat(batch_edge_query_relations, -1).long()
        return batch_edge_index, batch_edge_type, batch_h_positions, batch_t_positions, batch_query_relations, batch_edge_query_relations, batch_labels, num_ent

