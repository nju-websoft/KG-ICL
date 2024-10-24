import torch
import torch.nn as nn
from torch_scatter import scatter_add
from encoder.PromptEncoder import PromptEncoder
import numpy as np

import random

class GNNLayer(torch.nn.Module):
    def __init__(self, args, in_dim, out_dim, attn_dim, act=lambda x:x):
        super(GNNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act
        self.args = args

        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)
        self.W_h = nn.Linear(in_dim, out_dim, bias=False)
        self.msg_dropout = nn.Dropout(args.dropout)
        if torch.cuda.is_available() or not self.args.use_rspmm:
            from encoder.nbfnet_layers import GeneralizedRelationalConv
            self.conv = GeneralizedRelationalConv(input_dim=in_dim, output_dim=out_dim, query_input_dim=in_dim,
                                                  message_func="transe",
                                                  aggregate_func="sum", activation="relu", dependent=False,
                                                  project_relations=False)

        self.act = nn.RReLU()

    def forward(self, q_sub, q_rel, hidden, edges, n_node, rel_embeddings, qr_embeddings, loader):
        # edges:  [batch_idx, head, rela, tail, old_idx, new_idx]
        batch_idx = edges[:,0]
        sub = edges[:,4]
        rel = edges[:,2]
        obj = edges[:,5]

        hs = hidden[sub]
        hr = rel_embeddings.view(-1, self.in_dim)[rel + batch_idx * (loader.kg.relation_num + 1)]

        r_idx = edges[:, 0]
        h_qr = rel_embeddings[q_rel + torch.arange(q_rel.size(0)).to(self.args.device) * (loader.kg.relation_num + 1)][r_idx]

        alpha = self.w_alpha(
            nn.RReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr)))
        alpha = torch.sigmoid(alpha)  # [N_edge_of_all_batch, 1]

        # if not torch.cuda.is_available():
        if not torch.cuda.is_available() or not self.args.use_rspmm:
            message = hs + hr
            hidden = scatter_add(alpha * message, index=obj, dim=0, dim_size=n_node)
        else:
            edge_index__ = edges[:, [4, 5]]
            size = (sub.max() + 1, obj.max() + 1)
            hidden = self.msg_dropout(hidden)
            hidden = self.conv(relation=rel_embeddings, input=hidden, edge_index=edge_index__.t(),
                               edge_type=rel.view(-1) + batch_idx * (loader.kg.relation_num + 1),
                               size=size, edge_weight=alpha.view(-1))
        hidden = self.act(self.W_h(hidden))
        return hidden


class EntityEncoder(torch.nn.Module):
    def __init__(self, params):
        self.args = params
        super(EntityEncoder, self).__init__()
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x:x}
        act = acts[params.act]
        self.rel_dropout = nn.Dropout(self.args.dropout)

        self.relation_encoder = PromptEncoder(params)

        self.gnn_layers = nn.ModuleList([GNNLayer(self.args, self.hidden_dim, self.hidden_dim, self.attn_dim, act=act) for _ in range(self.n_layer)])
        self.rel_transfer = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.n_layer)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.n_layer)])
        self.layer_norms_rel = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.n_layer)])
        self.layer_norms_query = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.n_layer)])
        self.query_transfer = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.n_layer)])
        self.W_score = nn.ModuleList([nn.Linear(self.hidden_dim * 2, 1, bias=False) for _ in range(self.n_layer)])
       
        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)
        self.gate= nn.GRU(self.hidden_dim, self.hidden_dim)

    def forward(self, subs, rels, objs=None, loader=None, training=True, finetune=False):
        n = len(subs)
        # get unique relations to reduce computation
        unique_rels, unique_indices = np.unique(rels, return_inverse=True)
        n_ent = loader.kg.entity_num
        repeat_rels = np.repeat(np.expand_dims(unique_rels, 1), self.args.shot, axis=1).reshape(-1)

        if self.args.use_prompt_graph:
            edge_index, edge_type, h_positions, t_positions, query_relations, edge_query_relations, labels, num_ent = loader.get_case_graph(repeat_rels,
                                                                                                                     training and not finetune)
            rel_embeddings, query_embeddings, rel_embedding_full = self.relation_encoder(edge_index, edge_type, h_positions, t_positions, query_relations, edge_query_relations, labels,
                                                   num_ent, loader, shot=self.args.shot)
        else:
            rel_embeddings = torch.zeros((len(unique_rels), (loader.kg.relation_num*2+2), self.hidden_dim)).to(self.args.device)
            query_embeddings = None
            rel_embedding_full = None
            nn.init.xavier_uniform_(rel_embeddings)

        # recover the embeddings
        rel_embeddings = torch.index_select(rel_embeddings, 0, torch.from_numpy(unique_indices).to(self.args.device))
        rel_embeddings = rel_embeddings.reshape(-1, self.hidden_dim)

        q_sub = torch.from_numpy(subs).to(self.args.device).long()
        q_rel = torch.from_numpy(rels).to(self.args.device).long()

        nodes = torch.cat([torch.arange(n).unsqueeze(1).to(self.args.device), q_sub.unsqueeze(1)], 1)

        hidden = torch.index_select(rel_embeddings, 0, q_rel+torch.arange(q_rel.size(0)).to(self.args.device)*(loader.kg.relation_num+1))

        # dropout relations
        if self.args.use_augment:
            dropout_prob = self.args.relation_mask_rate
            dropout_prob = random.random() * dropout_prob
        else:
            dropout_prob = 0.0

        mask_batch = np.random.choice([0, 1], size=loader.kg.relation_num, p=[1 - dropout_prob, dropout_prob])

        distance2node = torch.ones((n, n_ent)).to(self.args.device)
        '''
        This setting aims to leverage the distance information between the target entity and the query entity. 
        If you're concerned about potential negative impacts on inference, 
        you can set the threshold as 0, which will have minimal effect on performance.
        '''
        if loader.kg.answer_distance[0] < 0.02:
            new_nodes = nodes
            distance2node[new_nodes[:, 0], new_nodes[:, 1]] = 0

        for i in range(self.n_layer):
            mask_layer = np.random.choice([0, 1], size=loader.kg.relation_num, p=[1 - dropout_prob, dropout_prob])
            mask_batch = mask_batch * mask_layer
            relation_list = np.arange(loader.kg.relation_num)
            mask_relations = relation_list[mask_batch == 1]

            nodes, edges, old_nodes_new_idx = loader.get_neighbors(nodes.data.cpu().numpy(), subs, rels, objs,
                                                                    mask_relations=mask_relations, training=training)

            if not training and loader.kg.answer_distance[i+1] < 0.02:
                new_nodes_new_idx = np.setdiff1d(np.arange(nodes.shape[0]), old_nodes_new_idx.detach().cpu().numpy())
                new_nodes = nodes[new_nodes_new_idx]
                distance2node[new_nodes[:, 0], new_nodes[:, 1]] = 0

            rel_embeddings_ = self.rel_transfer[i](rel_embeddings)
            rel_embeddings = rel_embeddings + torch.nn.functional.relu(rel_embeddings_)
            rel_embeddings = self.layer_norms_rel[i](rel_embeddings)

            hidden = self.gnn_layers[i](q_sub, q_rel, hidden, edges, nodes.size(0), rel_embeddings, query_embeddings, loader)
            if i == 0:
                h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).to(self.args.device)
            else:
                h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).to(self.args.device).index_copy_(1, old_nodes_new_idx, h0)
            hidden = self.dropout(hidden)
            hidden = self.layer_norms[i](hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)
            rel_embeddings = self.layer_norms_rel[i](rel_embeddings)

        scores = self.W_final(hidden).squeeze(-1)
        scores_all = torch.zeros((n, n_ent)).to(self.args.device)
        scores_all[[nodes[:,0], nodes[:,1]]] = scores
        scores_all = scores_all * distance2node
        return scores_all, rel_embedding_full, torch.from_numpy(repeat_rels).to(self.args.device)
