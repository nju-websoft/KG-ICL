import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch.nn.init import xavier_normal_


class PromptEncoder(torch.nn.Module):
    def __init__(self, params):
        super(PromptEncoder, self).__init__()
        self.args = params
        self.n_layer = params.n_relation_encoder_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim

        # initialize embeddings
        self.start_relation_embeddings = nn.Embedding(1, self.hidden_dim)
        self.position_embedding = nn.Embedding((self.args.path_hop+1)*(self.args.path_hop+1), self.hidden_dim)
        self.self_loop_embedding = nn.Embedding(1, self.hidden_dim)
        xavier_normal_(self.start_relation_embeddings.weight.data)
        xavier_normal_(self.position_embedding.weight.data)
        xavier_normal_(self.self_loop_embedding.weight.data)

        self.act = nn.RReLU()

        self.W_ht2r = nn.ModuleList([nn.Linear(self.hidden_dim * 3, self.hidden_dim, bias=True) for _ in range(self.n_layer)])
        self.W_message = nn.ModuleList([nn.Linear(self.hidden_dim * 3 if self.args.MSG == 'concat' else self.hidden_dim * 5, self.hidden_dim, bias=True) for _ in range(self.n_layer)])
        self.alpha = nn.ModuleList([nn.Linear(self.hidden_dim * 2, 1, bias=True) for _ in range(self.n_layer)])
        self.beta = nn.ModuleList([nn.Linear(self.hidden_dim * 2, 1, bias=True) for _ in range(self.n_layer)])
        self.loop_transfer = nn.ModuleList([nn.Linear(self.hidden_dim * 2, self.hidden_dim, bias=True) for _ in range(self.n_layer)])
        self.ent_transfer = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim, bias=True) for _ in range(self.n_layer)])
        self.rel_transfer = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim, bias=True) for _ in range(self.n_layer)])

        # readout
        self.final_to_rel_embeddings = nn.Linear(self.hidden_dim * self.n_layer, self.hidden_dim)

        # dropout and layer normalization
        self.dropout = nn.Dropout(0.3)
        self.layer_norm_rels = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.n_layer+1)])
        self.layer_norm_ents = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.n_layer+1)])
        self.layer_norm_loop = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.n_layer+1)])
        # self.gate_rel = nn.GRU(self.hidden_dim, self.hidden_dim)

    def forward(self, edge_index, edge_type, h_positions, t_positions, query_relations, edge_query_relations, labels, num_ent, loader, shot=5):
        relation_num = loader.kg.relation_num
        final_rel_embeddings = []

        if self.args.use_token_set:
            # initialize entity embeddings
            position = labels[:, 0] * (self.args.path_hop+1) + labels[:, 1]
            node_embeddings = torch.index_select(self.position_embedding.weight, 0, position)
            node_embeddings[h_positions] = self.position_embedding.weight[0]
            node_embeddings[t_positions] = self.position_embedding.weight[1]

            # initialize relation embeddings
            rel_embeddings = torch.zeros(relation_num * h_positions.size(-1), self.hidden_dim).cuda()
            rel_embeddings[query_relations+torch.arange(query_relations.size(-1)).cuda()*relation_num] = self.start_relation_embeddings.weight[0]
        else:
            # initialize entity and relation embeddings (w/o unified token set)
            node_embeddings = torch.zeros(num_ent, self.hidden_dim).cuda()
            rel_embeddings = torch.zeros(relation_num * h_positions.size(-1), self.hidden_dim).cuda()
            nn.init.xavier_uniform_(node_embeddings)
            nn.init.xavier_uniform_(rel_embeddings)

        self_loop_embeddings = self.self_loop_embedding.weight.unsqueeze(0).expand(
            rel_embeddings.size(0) // relation_num, -1, -1).reshape(-1, self.hidden_dim)

        # prompt encoder
        for i in range(self.n_layer):
            # 1. update node embeddings
            r_embeddings = torch.index_select(rel_embeddings, 0, edge_type)
            h_embeddings = torch.index_select(node_embeddings, 0, edge_index[0])
            q_embeddings = torch.index_select(rel_embeddings, 0, edge_query_relations)

            # 1.1 message with attention
            feature = self.entity_message(h_embeddings, r_embeddings, q_embeddings, self.args.MSG)
            message = self.act(self.W_message[i](feature))
            # alpha = self.entity_attention(torch.cat([r_embeddings, q_embeddings], dim=-1), edge_index,
            #                               node_embeddings.size(0), i)
            alpha = self.entity_attention(torch.cat([r_embeddings, q_embeddings], dim=-1), edge_index, node_embeddings.size(0), i)
            message = message * alpha
            ent_norm = self.compute_norm(edge_index, num_ent)
            message = message * ent_norm.view(-1, 1)

            # 1.2 aggregation
            node_embeddings = self.ent_aggregation(message, edge_index, num_ent, self.args.AGG, i)

            # 1.3 layer normalization
            node_embeddings = self.layer_norm_ents[i](node_embeddings)

            # 2. update relation embeddings
            h_embeddings = torch.index_select(node_embeddings, 0, edge_index[0])
            t_embeddings = torch.index_select(node_embeddings, 0, edge_index[1])
            r_embeddings = torch.index_select(rel_embeddings, 0, edge_type)

            # 2.1 message with attention
            feature = torch.cat([h_embeddings, t_embeddings, q_embeddings], dim=1)
            message = self.act(self.W_ht2r[i](feature))
            beta = self.relation_attention(torch.cat([r_embeddings, q_embeddings], dim=-1), edge_type,
                                           rel_embeddings.size(0), i)
            message = message * beta
            # 2.2 aggregation
            rel_embeddings = self.rel_aggregation(rel_embeddings, message, edge_type, rel_embeddings.size(0), i)
            # 2.3 layer normalization
            rel_embeddings = self.layer_norm_rels[i](rel_embeddings)

            # 3. store the relation embeddings of this layer
            final_rel_embeddings.append(rel_embeddings)

            # 4. update self-loop embeddings
            qr_embeddings = torch.index_select(rel_embeddings, 0, query_relations + torch.arange(
                query_relations.size(-1)).cuda() * relation_num)
            self_loop_embeddings = self_loop_embeddings + self.act(
                self.loop_transfer[i](torch.cat([self_loop_embeddings, qr_embeddings], dim=-1)))
            self_loop_embeddings = self.layer_norm_loop[i](self_loop_embeddings)

        # readout
        final_rel_embeddings = torch.cat(final_rel_embeddings, dim=-1)
        final_rel_embeddings = self.act(self.final_to_rel_embeddings(final_rel_embeddings))
        final_rel_embeddings = self.layer_norm_rels[-1](final_rel_embeddings)
        final_rel_embeddings = final_rel_embeddings.view(-1, shot, relation_num,  self.hidden_dim)

        # if multi-shot, then calculate the average of the final relation embeddings
        final_rel_embeddings_full = final_rel_embeddings
        final_rel_embeddings = torch.mean(final_rel_embeddings, dim=1).view(-1, relation_num, self.hidden_dim)
        self_loop_embeddings = torch.mean(self_loop_embeddings.view(-1, shot, 1, self.hidden_dim), dim=1).view(-1, 1, self.hidden_dim)
        final_rel_embeddings = self.dropout(final_rel_embeddings)

        final_rel_embeddings = final_rel_embeddings.view(-1, relation_num, self.hidden_dim)
        final_rel_embeddings = torch.cat([final_rel_embeddings, self_loop_embeddings], dim=1)
        return final_rel_embeddings, None, final_rel_embeddings_full

    def entity_message(self, h, r, q, MSG):
        if MSG == 'add':
            feature = h + r
        elif MSG == 'mul':
            feature = h * r
        elif MSG == 'concat':
            feature = torch.cat([h, r, q], dim=-1)
        elif MSG == 'mix':
            feature = torch.cat([h * r, h + r, h, r, q], dim=-1)
        else:
            raise NotImplementedError
        return feature

    def entity_attention(self, feature, edge_index, num_nodes, i):
        if self.args.use_attn:  # for ablation study
            alpha = self.alpha[i](self.act(feature))
            if self.args.attn_type == 'GAT':
                alpha = torch.exp(alpha)
                alpha = alpha / (torch.index_select(
                    scatter_add(alpha, edge_index[1], dim=0, dim_size=num_nodes)[edge_index[1]] + 1e-10,
                    0, edge_index[1]))
            elif self.args.attn_type == 'Sigmoid':
                alpha = torch.sigmoid(alpha)
        else:
            alpha = 1.0
        return alpha

    def ent_aggregation(self, message, edge_index, num_ent, AGG, i):
        ent_norm = self.compute_norm(edge_index, num_ent)
        message = message * ent_norm.view(-1, 1)
        if AGG == 'sum':
            node_embeddings_ = scatter_add(message, index=edge_index[1], dim=0, dim_size=num_ent)
        elif AGG == 'max':
            node_embeddings_, _ = scatter_max(message, index=edge_index[1], dim=0, dim_size=num_ent)
        elif AGG == 'mean':
            node_embeddings_ = scatter_mean(message, index=edge_index[1], dim=0, dim_size=num_ent)
        else:
            raise NotImplementedError
        node_embeddings_ = self.act(self.ent_transfer[i](node_embeddings_))
        return node_embeddings_

    def relation_attention(self, feature, edge_type, num_rels, i):
        if self.args.use_attn:
            beta = self.beta[i](feature)
            if self.args.attn_type == 'GAT':
                beta = torch.exp(beta)
                beta = beta / (torch.index_select(
                    scatter_add(beta, index=edge_type, dim=0, dim_size=num_rels)[edge_type] + 1e-10, 0,edge_type))
            elif self.args.attn_type == 'Sigmoid':
                beta = torch.sigmoid(beta)
            else:
                raise NotImplementedError
        else:
            beta = 1.0
        return beta

    def rel_aggregation(self, rel_embeddings, message, edge_type, num_rels, i):
        if self.args.AGG_rel == 'max':
            rel_embeddings_, _ = scatter_max(message, index=edge_type, dim=0,
                                             dim_size=num_rels)
        elif self.args.AGG_rel == 'sum':
            rel_embeddings_ = scatter_add(message, index=edge_type, dim=0,
                                          dim_size=num_rels)
        elif self.args.AGG_rel == 'mean':
            rel_embeddings_ = scatter_mean(message, index=edge_type, dim=0,
                                           dim_size=num_rels)
        else:
            raise NotImplementedError
        rel_embeddings_ = self.act(self.rel_transfer[i](rel_embeddings_))
        rel_embeddings = rel_embeddings_ + rel_embeddings
        rel_embeddings = rel_embeddings.squeeze(0)
        return rel_embeddings

    def compute_norm(self, edge_index, num_ent):
        col, row = edge_index
        edge_weight = torch.ones_like(row).float()
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent)  # Summing number of weights of the edges
        deg_inv = deg.pow(-0.5)  # D^{-0.5}
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]  # D^{-0.5}
        return norm