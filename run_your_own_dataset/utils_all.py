import networkx as nx
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
import random, os
from tqdm import tqdm
import scipy.sparse as ssp
import torch
from torch_scatter import scatter_add
import json

# read data
def read_triple(file_path, triple_format):
    # return triples
    with open(file_path, 'r', encoding='utf-8') as f:
        line = f.readline()
        triples = []
        while line:
            l = line.strip().split('\t')
            if len(l) < 3:
                line = f.readline()
                print(line)
                continue
            if triple_format == 'hrt':
                triples.append((l[0], l[1].replace('/', '#'), l[2]))
            elif triple_format == 'htr':
                triples.append((l[0], l[2].replace('/', '#'), l[1]))
            elif triple_format == 'rht':
                triples.append((l[1], l[0].replace('/', '#'), l[2]))
            line = f.readline()
    return list(set(triples))


def get_entities_relations_from_triples(triples):
    # return entities, relations
    entities = set()
    relations = set()
    for triple in triples:
        entities.add(triple[0])
        entities.add(triple[2])
        relations.add(triple[1])
    return entities, relations


def set2dict(s):
    # return dict
    d, d_inv = {}, {}
    for i, e in enumerate(s):
        d[e] = i
        d_inv[i] = e
    return d, d_inv


def read_triples_from_json(file_path):
    rel2triples = json.load(open(file_path, 'r'))
    triples = []
    for rel in rel2triples:
        triples_ = rel2triples[rel]
        for triple in triples_:
            # print((triple[0], rel, triple[2]))
            triples.append((triple[0], rel, triple[2]))
    return triples

def read_triples_from_hr2t_json(file_path):
    hr2t = json.load(open(file_path, 'r'))
    triples = []
    for hr in hr2t:
        try:
            _, h_, r_ = hr.split('concept:')
            h, r = 'concept:' + h_, 'concept:' + r_
        except:
            h, r_ = hr.split('P')
            r = 'P' + r_
        triples_ = hr2t[hr]
        for ent in triples_:
            triples.append((h, r, ent))
    return triples


def read_triples_from_list(file_path):
    triples_ = json.load(open(file_path, 'r'))
    triples = []
    for triple in triples_:
        triples.append((triple[0], triple[1], triple[2]))
    return triples


def sample_example_from_triples_by_relation(triples, sample_num):
    # return examples
    examples = []
    rel2triples = defaultdict(lambda: set())
    for triple in triples:
        rel2triples[triple[1]].add(triple)
    for rel in rel2triples:
        if sample_num > len(rel2triples[rel]):
            print(rel)
        examples.extend(random.sample(rel2triples[rel], min(sample_num, len(rel2triples[rel]))))
    return examples


def read_candidate_for_few_shot(file_path, rel2id, ent2id):
    rel2triples = json.load(open(file_path, 'r'))
    rel2candidate_ids = defaultdict(lambda: list())
    for rel_ in rel2triples.keys():
        # print(rel2triples[rel])
        rel = int(rel2id[rel_])
        for ent in rel2triples[rel_]:
            # print(triple)
            rel2candidate_ids[rel].append(int(ent2id[ent]))
    return rel2candidate_ids


def write_candidate_for_few_shot(file_path, rel2triples_ids):
    print(len(rel2triples_ids))
    with open(file_path, 'w', encoding='utf-8') as f:
        # print(rel2triples_ids)
        json.dump(rel2triples_ids, f)


def triple2ids(triples, entity2id, relation2id):
    # return triples_ids
    triples_ids = []
    for triple in triples:
        triples_ids.append((entity2id[triple[0]], relation2id[triple[1]], entity2id[triple[2]]))
    return triples_ids


def get_ht2r(triples):
    ht2r = defaultdict(lambda: set())
    for triple in triples:
        h, r, t = triple
        ht2r[(h, t)].add(r)
    return ht2r


def get_rel2triples(triples):
    rel2triples = defaultdict(lambda: set())
    for triple in triples:
        h, r, t = triple
        rel2triples[r].add(triple)
    for rel in rel2triples:
        rel2triples[rel] = list(rel2triples[rel])
    return rel2triples


# write data
def write_triple(file_path, triples):
    # write triples to file_path
    with open(file_path, 'w', encoding='utf-8') as f:
        for triple in triples:
            # print(triple)
            f.write(str(triple[0]) + '\t' + str(triple[1]) + '\t' + str(triple[2]) + '\n')


def write_list(file_path, l):
    # write list to file_path
    with open(file_path, 'w', encoding='utf-8') as f:
        for e in l:
            f.write(e + '\n')


def write_dict(file_path, dic):
    # write dict to file_path
    with open(file_path, 'w', encoding='utf-8') as f:
        for key, value in dic.items():
            f.write(str(key) + '\t' + str(value) + '\n')


def write_cases(file_path, cases):
    # print(cases)
    # 每个case包括两部分，分别是triples和labels，写入两个文件，分别是triples.txt和labels.txt
    with open(file_path + '.triples', 'w', encoding='utf-8') as f:
        # print('-----', cases)
        for case in cases[:-2]:
            # print(case)
            for triple in case:
                f.write(str(triple[0]) + '\t' + str(triple[1]) + '\t' + str(triple[2]) + '\n')
            f.write('\n')
    with open(file_path + '.labels', 'w', encoding='utf-8') as f:
        labels = cases[-2]
        for ent, label in labels.items():
            f.write(str(ent) + '\t' + str(label[0]) + '\t' + str(label[1]) + '\n')
    with open(file_path + '.node2id', 'w', encoding='utf-8') as f:
        node2id = cases[-1]
        # 将node2id按照keys排序
        node2id = sorted(node2id.items(), key=lambda x: x[0])
        for node, id in node2id:
            f.write(str(node) + '\t' + str(id) + '\n')


def write_matrix(file_path, matrix):
    # write matrix using numpy
    np.save(file_path, matrix)


'''build cases'''
# 获取csc矩阵的第col列有效数据
def get_column(matrix, col):
    data = matrix.data[matrix.indptr[col]:matrix.indptr[col + 1]]
    row = matrix.indices[matrix.indptr[col]:matrix.indptr[col + 1]]
    return row, data


class KG:
    def __init__(self, triples, n_ent, n_rel):
        self.triples = triples
        self.n_rel = n_rel
        self.n_ent = n_ent

    def build_cases_for_large_graph(self, case_num, enclosing=True, hop=3, EA=False, query_relations=None):
        # prepare
        self.ht2r = get_ht2r(self.triples)
        self.build_nx_graph()
        rel2triples = get_rel2triples(self.triples)
        # build cases
        self.rel2cases = {rel: list() for rel in range(self.n_rel)}
        # print(111)
        if query_relations is None:
            query_relations = range(self.n_rel)
        for rel in tqdm(query_relations):
            if EA:
                # print(222)
                if rel != self.n_rel - 1:
                    # print(rel)
                    continue
            all_index = [i for i in range(len(rel2triples[rel]))]
            if len(all_index) < min(case_num * 3, len(rel2triples[rel])):
                cases_triples_idx = all_index
            else:
                cases_triples_idx = np.random.choice(all_index, min(case_num * 3, len(rel2triples[rel])), replace=False)
            cases_triples = [rel2triples[rel][i] for i in cases_triples_idx]
            for triple in cases_triples:
                case = self.build_subgraph_large(self.G, triple, enclosing=enclosing, hop=hop)
                if len(case[1]) > 0:
                    self.rel2cases[rel].append(case)
                if len(self.rel2cases[rel]) >= case_num:
                    break
        # print(self.rel2cases)
        return self.rel2cases

    def build_nx_graph(self):
        self.G = nx.Graph()
        all_triples = [(triple[0], triple[2]) for triple in self.triples] + [(triple[2], triple[0]) for triple in
                                                                             self.triples]
        self.G.add_edges_from(all_triples)

    def build_subgraph_large(self, G, triple, enclosing=True, hop=3):
        head, query_relation, tail = triple
        labels, enclosing_subgraph_nodes = self.get_enclosing_nodes_large(head, tail, enclosing=enclosing, hop=hop)
        if head != tail:
            nodes2id = {node: i + 2 for i, node in enumerate(set(enclosing_subgraph_nodes) - {head, tail})}
            nodes2id[head] = 0
            nodes2id[tail] = 1
        else:
            nodes2id = {node: i + 1 for i, node in enumerate(set(enclosing_subgraph_nodes) - {head})}
            nodes2id[head] = 0
        # 从networkx的图G中取出enclosing_subgraph_nodes的子图的edges以及对应的relation
        # print('---enclosing_subgraph_nodes', enclosing_subgraph_nodes)
        edges_index_ = nx.subgraph(G, enclosing_subgraph_nodes).edges()
        # print('---edges_index_', edges_index_)
        edges_index = []
        edges_type = []
        for edge in edges_index_:
            rels = self.ht2r[(edge[0], edge[1])]
            edges_type += list(rels)
            edges_index += [(edge[0], edge[1]) for i in range(len(rels))]
            # 添加反向边
            rels = self.ht2r[(edge[1], edge[0])]
            edges_type += list(rels)
            edges_index += [(edge[1], edge[0]) for i in range(len(rels))]
        edges_index = np.array(edges_index).T
        edges_type = np.array(edges_type)

        # return edges_index, edges_type, query_relation, labels, len(enclosing_subgraph_nodes)
        if len(edges_index) == 0:
            return [], {}
        triples = [[nodes2id[triple[0]], triple[1], nodes2id[triple[2]]]]

        for i in range(len(edges_type)):
            if [nodes2id[edges_index[0][i]], edges_type[i], nodes2id[edges_index[1][i]]] != [nodes2id[triple[0]], triple[1], nodes2id[triple[2]]]:
                triples.append([nodes2id[edges_index[0][i]], edges_type[i], nodes2id[edges_index[1][i]]])
        ent2label = {nodes2id[node]: labels[i] for i, node in enumerate(set(enclosing_subgraph_nodes).union({head, tail}))}
        ent_subid2ent_id = {nodes2id[node]: node for i, node in enumerate(set(enclosing_subgraph_nodes).union({head, tail}))}
        return triples, ent2label, ent_subid2ent_id

    def get_3hop_neighbors(self, node):
        A = self.matrix_3hop_enclosing
        row, data = get_column(A, node)
        return set(row.tolist())

    def get_enclosing_nodes_large(self, head, tail, enclosing=True, all=False, hop=3):
        root1_nei, head_dict = get_3hop_neighbors_with_distances(self.G, head, hop=hop)
        root2_nei, tail_dict = get_3hop_neighbors_with_distances(self.G, tail, hop=hop)

        if not all:
            subgraph_nei_nodes = set(root1_nei).intersection(set(root2_nei)) - {head, tail}
        else:
            subgraph_nei_nodes = root1_nei.union(root2_nei) - {head, tail}

        # if not enclosing:
        #     root1_nei_close = get_3hop_neighbors_with_distances(self.G, head, hop=hop)[0]
        #     root2_nei_close = get_3hop_neighbors_with_distances(self.G, tail, hop=hop)[0]
        #     limit_num = 50
        #     root1_nei_1hop = set(root1_nei).intersection(root1_nei_close)
        #     root2_nei_2hop = set(root2_nei).intersection(root2_nei_close)
        #     if len(root1_nei_close) > limit_num:
        #         root1_nei_close = set(random.sample(root1_nei_close, limit_num))
        #     if len(root2_nei_close) > limit_num:
        #         root2_nei_close = set(random.sample(root2_nei_close, limit_num))
        #
        #
        # self.G.add_edges_from([(head, tail), (tail, head)])

        subgraph_nodes = list([head, tail]) + sorted(list(subgraph_nei_nodes))

        max_distance = hop

        dist2ht = []
        for node in subgraph_nodes:
            dist2h = head_dict[node] if node in head_dict else 10000000
            dist2t = tail_dict[node] if node in tail_dict else 10000000
            dist2ht.append([dist2h, dist2t])
        if head != tail:
            dist2ht[0] = [0, 1]
            dist2ht[1] = [1, 0]
        else:
            dist2ht[0] = [0, 0]
        labels = np.array(dist2ht)

        if all:
            labels_ = [labels[i] for i in subgraph_nodes]
            return labels_, subgraph_nodes

        if not enclosing:
            close_cond = np.sum(labels, axis=1) <= max_distance
            open_cond = np.max(labels, axis=1) <= max_distance
            # 从open_cond 但非 close_cond中的为True的随机选最多50个节点，加上所有的close_cond，得到最终的conditions(bool)
            open_not_close = np.where(open_cond & ~close_cond)[0]
            conditions = np.zeros(len(labels), dtype=bool)
            if len(open_not_close) > 50:
                open_not_close = np.random.choice(open_not_close, 50, replace=False)
            conditions[open_not_close] = True
            conditions[close_cond] = True
            enclosing_subgraph_nodes = [subgraph_nodes[i] for i in np.where(conditions)[0]]
            labels = labels[np.where(conditions)[0]]
        else:
            conditions = np.sum(labels, axis=1) <= max_distance
            enclosing_subgraph_nodes = [subgraph_nodes[i] for i in
                                        np.where(conditions)[0]]
            labels = labels[np.where(conditions)[0]]
        return labels, enclosing_subgraph_nodes

    def sample_train_data_random(self, num=100):
        # 从训练集中随机采样num个三元组
        train_data = self.triples
        train_data = np.array(train_data)
        train_data = train_data[np.random.choice(len(train_data), num, replace=False)]
        return train_data

    def sample_train_data_by_relation(self, num=100, triples=None):
        if triples is None:
            train_data = self.triples
        else:
            train_data = triples
        rel2train = defaultdict(list)
        for triple in train_data:
            rel2train[triple[1]].append(triple)
        train_data = []
        for rel in rel2train:
            if len(rel2train[rel]) < num:
                train_data += rel2train[rel]
                continue
            train_data += random.sample(rel2train[rel], num)
        return train_data


def get_3hop_neighbors_with_distances(graph, root_node, hop=3):
    distances = nx.single_source_shortest_path_length(graph, root_node, cutoff=hop)
    # print(distances)

    # 过滤出三跳邻居的距离信息
    three_hop_distances = {node: distance for node, distance in distances.items()}

    # return three_hop_neighbors, three_hop_distances
    return list(three_hop_distances.keys()), three_hop_distances


import pickle
def read_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data




