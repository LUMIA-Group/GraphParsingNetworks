from torch_geometric.datasets import TUDataset
from sklearn.model_selection import KFold
import torch
import os

# use 10 fold cross-validation
def k_fold(folds, seed, labels):
    kf = KFold(folds, shuffle=True, random_state=seed)

    test_indices, train_indices = [], []
    for _, idx in kf.split(torch.zeros(len(labels)), labels):
        test_indices.append(torch.from_numpy(idx))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(labels), dtype=torch.uint8)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices

for dataset_name in ["FRANKENSTEIN", "NCI109", "NCI1", "PROTEINS", "DD"]:
    for seed in range(20):
        dataset = TUDataset("../pyg", dataset_name, use_node_attr=True, cleaned=False)
        train_indices, test_indices, val_indices = k_fold(folds=10, seed=seed, labels=dataset.data.y)
        os.makedirs("datasets/%s/10fold_idx_%s"%(dataset_name, seed))

        for fold in range(len(train_indices)):
            with open("datasets/%s/10fold_idx_%s/train_idx-%s.txt"%(dataset_name, seed, fold), 'w') as f:
                for idx in train_indices[fold].tolist():
                    f.write(str(idx)+'\n')

        for fold in range(len(test_indices)):
            with open("datasets/%s/10fold_idx_%s/test_idx-%s.txt"%(dataset_name, seed, fold), 'w') as f:
                for idx in test_indices[fold].tolist():
                    f.write(str(idx)+'\n')

        for fold in range(len(val_indices)):
            with open("datasets/%s/10fold_idx_%s/val_idx-%s.txt"%(dataset_name, seed, fold), 'w') as f:
                for idx in val_indices[fold].tolist():
                    f.write(str(idx)+'\n')