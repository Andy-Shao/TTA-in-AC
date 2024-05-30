from typing import Any
import random
import pandas as pd

import matplotlib.pyplot as plt

import torch 
from torch.utils.data import Dataset

class SubsetDs(Dataset):
    def __init__(self, dataset: Dataset, indexes) -> None:
        super().__init__()
        self.dataset = dataset
        self.indexes = indexes

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index) -> Any:
        return self.dataset[self.indexes[index]]
    
def calIndexes(dataset: Dataset, n_flod: int) -> pd.DataFrame:
    indexes = pd.DataFrame(columns=['index', 'fold'])
    for index in range(len(dataset)):
        fold = random.randrange(start=0, stop=n_flod)
        indexes.loc[len(indexes)] = [index, fold]
    return indexes

def switchFold(val_fold: int, indexes: pd.DataFrame):
    train_indexes = indexes[indexes['fold'] != val_fold]['index'].to_numpy()
    val_indexes = indexes[indexes['fold'] == val_fold]['index'].to_numpy()
    return (train_indexes, val_indexes)
    
class ValidationRecord():
    def __init__(self, n_fold: int, label_size: int, n_iter=1) -> None:
        self.records = pd.DataFrame(columns=['iteration','val_fold', 'label', 'accuracy', 'precision', 'recall', 'TP', 'FP', 'FN', 'TN'])
        cols = [f'pred{i}' for i in range(label_size)]
        cols.append('label')
        cols.append('iteration')
        cols.append('fold')
        self.vLog = pd.DataFrame(columns=cols)
        self.n_iter = n_iter
        self.n_fold = n_fold
        self.label_size = label_size
        for iter in range(n_iter):
            for fold_id in range(n_fold):
                for label in range(label_size):
                    self.records.loc[len(self.records)] = [iter, fold_id, label, 0.0, 0.0, 0.0, 0, 0, 0, 0]
                self.records.loc[len(self.records)] = [iter, fold_id, -1 , 0.0, 0.0, 0.0, 0, 0, 0, 0] ## ttl line

    def noteRecord(self, outputs: torch.Tensor, labels: torch.Tensor, val_fold: int, iter=0):
        """
        :param outputs: it is one-hot vector
        :param labels: it is one-hot vector
        """
        
        _, preds = torch.max(input=outputs, dim=1)
        _, class_ids = torch.max(input=labels, dim=1)
        logs = torch.cat((outputs.cpu(), class_ids.unsqueeze(1).cpu(), torch.ones((outputs.shape[0], 1)) * iter, torch.ones((outputs.shape[0], 1)) * val_fold), dim=1).numpy()
        for log in logs:
            self.vLog.loc[len(self.vLog)] = log
        preds = preds.cpu().numpy()
        class_ids = class_ids.cpu().numpy()
        for i, class_id in enumerate(class_ids):
            if preds[i] == class_ids[i]: 
                rc = self.records[self.records['val_fold'] == val_fold]
                rc = rc[rc['iteration'] == iter]
                for k, row in rc.iterrows():
                    if row['label'] == class_ids[i]:
                        self.records.loc[k, 'TP'] += 1
                    else:
                        self.records.loc[k, 'TN'] += 1
            else: 
                rc = self.records[self.records['val_fold'] == val_fold]
                rc = rc[rc['iteration'] == iter]
                for k, row in rc.iterrows():
                    if row['label'] == class_ids[i]:
                        self.records.loc[k, 'FN'] += 1
                    elif row['label'] == preds[i]:
                        self.records.loc[k, 'FP'] += 1
                    else:
                        self.records.loc[k, 'TN'] += 1

    def getRecord(self):
        return self.records.copy(deep=True)

    def getValidateLog(self):
        return self.vLog

    def calRecord(self):
        # calculate each label
        for i, row in self.records[self.records['label'] != -1].iterrows():
            TP = row['TP']
            FP = row['FP']
            FN = row['FN']
            TN = row['TN']
            self.records.loc[i, 'accuracy'] = TP / (TP + FP)
            self.records.loc[i, 'precision'] = ValidationRecord.precision(TP, FP)
            self.records.loc[i, 'recall'] = ValidationRecord.recall(TP, FN)         

        # calculate ttl
        for i, row in self.records[self.records['label'] == -1].iterrows():
            val_fold = row['val_fold']
            iteration = row['iteration']
            TP = 0
            FP = 0
            TN = row['TN']
            FN = 0
            rc = self.records[self.records['iteration'] == iteration]
            rc = rc[rc['val_fold'] == val_fold]
            rc = rc[rc['label'] != -1]
            for k, innerow in rc.iterrows():
                TP += innerow['TP']
                FP += innerow['FP']
                FN += innerow['FN']
                
            self.records.loc[i, 'precision'] = ValidationRecord.precision(TP, FP)
            self.records.loc[i, 'recall'] = ValidationRecord.recall(TP, FN)
            self.records.loc[i, 'accuracy'] = TP / TN

    @staticmethod
    def precision(TP: int, FP: int):
        return TP / (TP + FP)
    
    @staticmethod
    def recall(TP: int, FN: int):
        return TP / (TP + FN)
    
def corss_validation_analysis(
        dataset: Dataset, n_fold: int, label_size: int, tranFunc, inferFunc, 
        n_iter=1
    ) -> ValidationRecord:
    records = ValidationRecord(n_fold=n_fold, n_iter=n_iter, label_size=label_size)
    for iter in range(n_iter):
        indexes = calIndexes(dataset=dataset, n_flod=n_fold)
        for val_fold in range(n_fold):
            train_indexes, val_indexes = switchFold(val_fold=val_fold, indexes=indexes)
            model = tranFunc(dataset, train_indexes)
            inferFunc(model, iter, val_fold, dataset, val_indexes, records)

    return records

def display_confidence_interval(label_id: int, input_vector, z_score=1.96, color='#2187bb', end_point_width=.25):
    input_tensor = torch.Tensor(input_vector)
    mean = input_tensor.mean().item()
    std = input_tensor.std().item()
    confidence_interval = z_score * std / torch.sqrt(torch.Tensor([input_tensor.shape[0]])).item()

    left = label_id - end_point_width / 2
    right = label_id + end_point_width / 2
    top = mean + confidence_interval
    bottom = mean - confidence_interval
    
    plt.plot([label_id, label_id], [bottom, top], color=color)
    plt.plot([left, right], [bottom, bottom], color=color)
    plt.plot([left, right], [top, top], color=color)
    plt.plot(label_id, mean, 'ro')

    return mean, confidence_interval