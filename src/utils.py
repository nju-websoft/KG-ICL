import numpy as np
from scipy.stats import rankdata
import subprocess
import logging
import networkx as nx
from collections import defaultdict
import torch


def cal_ranks(scores, labels, filters):
    scores = scores - np.min(scores, axis=1, keepdims=True) + 1e-8
    full_rank = rankdata(-scores, method='ordinal', axis=1)
    filter_scores = scores
    # We fixed a bug in RED-GNN to ensure fairness.
    # In case of a tie, the results will be sorted according to their ID order.
    filter_scores[filters == 0] = -10000000
    filter_rank = rankdata(-filter_scores, method='ordinal', axis=1)
    ranks = (full_rank - filter_rank + 1) * labels
    ranks = ranks[np.nonzero(ranks)]
    return list(ranks)


def cal_performance(ranks):
    mrr = (1. / ranks).sum() / len(ranks)
    h_1 = sum(ranks<=1) * 1.0 / len(ranks)
    h_3 = sum(ranks<=3) * 1.0 / len(ranks)
    h_5 = sum(ranks<=5) * 1.0 / len(ranks)
    h_10 = sum(ranks<=10) * 1.0 / len(ranks)
    return mrr, h_1, h_3, h_5, h_10


def get_distance(facts, background):
    G = nx.Graph()
    for triple in background:
        G.add_edge(triple[0], triple[2])
    distance = []
    un_reachable = 0
    for fact in facts:
        try:
            distance.append(nx.shortest_path_length(G, source=fact[0], target=fact[2]))
        except:
            un_reachable += 1
    distance = np.array(distance)
    dist2rate = defaultdict(float)
    for i in range(10):
        dist2rate[i] = np.sum(distance == i) / len(distance)
    return dist2rate



