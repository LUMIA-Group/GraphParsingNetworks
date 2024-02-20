import torch
import numpy as np
from functools import reduce
from utils.data_asap import get_dataset
from torch_geometric.loader import DataLoader

class Trainer(object):

    def __init__(self, args):

        super(Trainer, self).__init__()

        self.args = args
        self.dataset = self.load_data()
        self.train_loader, self.val_loader, self.test_loader = self.load_dataloader()

    def load_data(self):

        dataset = get_dataset(self.args.data, normalize=self.args.normalize)
        self.args.num_features, self.args.num_classes, self.args.avg_num_nodes = dataset.num_features, dataset.num_classes, np.ceil(np.mean([data.num_nodes for data in dataset]))
        print('# %s: [FEATURES]-%d [NUM_CLASSES]-%d [AVG_NODES]-%d' % (dataset, self.args.num_features, self.args.num_classes, self.args.avg_num_nodes))

        return dataset

    def load_dataloader(self):
        fold_number = self.args.fold_number

        train_idxes = torch.as_tensor(np.loadtxt('datasets/asap/datasets/%s/10fold_idx_%s/train_idx-%d.txt' % (self.args.data, self.args.seed, fold_number),
                                                dtype=np.int32), dtype=torch.long)
        test_idxes = torch.as_tensor(np.loadtxt('datasets/asap/datasets/%s/10fold_idx_%s/test_idx-%d.txt' % (self.args.data, self.args.seed, fold_number),
                                                dtype=np.int32), dtype=torch.long)
        val_idxes = torch.as_tensor(np.loadtxt('datasets/asap/datasets/%s/10fold_idx_%s/val_idx-%d.txt' % (self.args.data, self.args.seed, fold_number),
                                                dtype=np.int32), dtype=torch.long)

        all_idxes = reduce(np.union1d, (train_idxes, val_idxes, test_idxes))
        assert len(all_idxes) == len(self.dataset)

        train_set, val_set, test_set = self.dataset[train_idxes], self.dataset[val_idxes], self.dataset[test_idxes]

        train_loader = DataLoader(dataset=train_set, batch_size=self.args.batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_set, batch_size=self.args.batch_size, shuffle=False)
        test_loader = DataLoader(dataset=test_set, batch_size=self.args.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

class Args():
    pass

def get_dataloaders(task, seed, fold_number, batch_size):
    args = Args()
    args.normalize = False
    args.data = task
    args.seed = seed
    args.fold_number = fold_number+1
    args.batch_size = batch_size

    trainer = Trainer(args)

    return trainer.train_loader, trainer.val_loader, trainer.test_loader, trainer.args.num_features, trainer.args.num_classes