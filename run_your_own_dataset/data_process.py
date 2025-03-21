import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils_all import *
import shutil


def data_progress(dataset, triple_format='htr'):
    # 1. read data
    file_list = ['train', 'valid', 'test']
    triples_dict = {file: [] for file in file_list}
    for file in file_list:
        triples_dict[file] = read_triple(os.path.join(dataset, file + '.txt'), triple_format=triple_format)
    all_triples = triples_dict['train'] + triples_dict['valid'] + triples_dict['test']
    all_triples = list(set(all_triples))
    entities, relations = get_entities_relations_from_triples(all_triples)
    entity2id, id2entity = set2dict(entities)
    relation2id, id2relation = set2dict(relations)
    for file in file_list:
        triples_dict[file] = triple2ids(triples_dict[file], entity2id, relation2id)
    valid_filter_triples = triples_dict['train'] + triples_dict['valid']
    test_filter_triples = triples_dict['train'] + triples_dict['valid'] + triples_dict['test']

    # 2. build cases
    kg = KG(triples_dict['train'], len(entity2id), len(relation2id))
    cases = kg.build_cases_for_large_graph(case_num=25, enclosing=False, hop=3)
    # 3. sample training_data
    train_data = kg.sample_train_data_by_relation(num=200)

    # 4. write data
    for file in file_list:
        if not os.path.exists(os.path.join(output_dir, dataset, file)):
            os.makedirs(os.path.join(output_dir, dataset, file))
        else:
            shutil.rmtree(os.path.join(output_dir, dataset, file))
            os.makedirs(os.path.join(output_dir, dataset, file))

        if file in ['train', 'valid']:
            write_triple(os.path.join(output_dir, dataset, file, 'background.txt'), triples_dict['train'])
        else:
            write_triple(os.path.join(output_dir, dataset, file, 'background.txt'), triples_dict['train'] + triples_dict['valid'])
        # entity2id.txt = entity2id
        write_dict(os.path.join(output_dir, dataset, file, 'entity2id.txt'), entity2id)
        # relation2id.txt = relation2id
        write_dict(os.path.join(output_dir, dataset, file, 'relation2id.txt'), relation2id)
        if file == 'train':
            try:
                write_triple(os.path.join(output_dir, dataset, file, 'facts.txt'), train_data)
            except:
                write_triple(os.path.join(output_dir, dataset, file, 'facts.txt'), triples_dict[file])
        else:
            write_triple(os.path.join(output_dir, dataset, file, 'facts.txt'), triples_dict[file])
        if file in ['train', 'valid']:
            write_triple(os.path.join(output_dir, dataset, file, 'filter.txt'), valid_filter_triples)
        else:
            write_triple(os.path.join(output_dir, dataset, file, 'filter.txt'), test_filter_triples)
        # store cases
        if not os.path.exists(os.path.join(output_dir, dataset, file, 'cases')):
            os.makedirs(os.path.join(output_dir, dataset, file, 'cases'))
        for relation in relation2id.keys():
            if not os.path.exists(os.path.join(output_dir, dataset, file, 'cases', relation)):
                os.makedirs(os.path.join(output_dir, dataset, file, 'cases', relation))
        for relation in relation2id.keys():
            if cases[relation2id[relation]] is None or len(cases[relation2id[relation]]) == 0:
                continue
            for i in range(len(cases[relation2id[relation]])):
                write_cases(os.path.join(output_dir, dataset, file, 'cases', relation, str(i)), cases[relation2id[relation]][i])


dataset_list = ['your_dataset']

output_dir = '../processed_data/'
for dataset in dataset_list:
    print(dataset)
    if dataset in ['ConceptNet']:
        data_progress(dataset, triple_format='rht')
    else:
        data_progress(dataset, triple_format='hrt')



