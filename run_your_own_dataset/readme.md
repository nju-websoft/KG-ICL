# Run your dataset

This guide will help you preprocess your own dataset and complete subsequent evaluation steps. Please follow the instructions below.

---

## Dataset Preprocessing Steps

### 1. Prepare Your Dataset
You need to prepare the following three files:
- `your_dataset/train.txt`
- `your_dataset/valid.txt`
- `your_dataset/test.txt`

These files should be placed in the current directory.

### 2. File Format Requirements
Each line in these files should contain the following content:
- `subject_entity relation object_entity`

Please use '\t' to separate the three elements.

### 3. Preprocess Your Dataset
Open the `data_process.py`, and replace the **dataset_list** with the name of your dataset.
Run the following command to complete the preprocessing.

    python data_process.py

## Inference on Your Dataset
After completing the preprocessing, navigate to the src directory and run the following command:

    python evaluation.py --checkpoint_path ../checkpoint/KG-ICL-6L --test_dataset_list your_dataset


