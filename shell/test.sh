#!/bin/bash
path=$(pwd)

echo $path
PTmodels=(5)
base_path="../checkpoint/KG-ICL-6L"
test_datasets3=('WN18RR_v1_ind' 'WN18RR_v2_ind' 'WN18RR_v3_ind' 'WN18RR_v4_ind' 'fb237_v1_ind' 'fb237_v2_ind' 'fb237_v3_ind' 'fb237_v4_ind' 'nell_v1_ind' 'nell_v2_ind' 'nell_v3_ind' 'nell_v4_ind' 'FB-25' 'FB-50' 'FB-75' 'FB-100' 'NL-0' 'NL-25' 'NL-50' 'NL-75' 'NL-100' 'WK-25' 'WK-50' 'WK-75' 'WK-100' 'WN18RR' 'NELL-995' 'FB15k-237' 'ConceptNet' 'codex-s' 'codex-m' 'codex-l' 'YAGO3-10' 'WD-singer' 'NELL23K' 'FB15K-237-10' 'FB15K-237-20' 'FB15K-237-50' 'ILPC-large' 'ILPC-small' 'DB100K' 'aristo-v4' 'Hetionet')

for dataset in "${test_datasets3[@]}"; do
  model_name=$base_path
  echo "Running evaluation for dataset: $dataset"
  python ../src/evaluation.py \
          --checkpoint_path "$model_name" \
          --test_dataset_list "$dataset" \
          --gpu 0 \
          --n_layer 6 \
          --hidden_dim 32 \
          --MSG 'concat'\
          --attn_dim 5 \
          --shot 5 \
          --act 'idd' \
          --note ""
  echo "-----------------------------------------------"
done
