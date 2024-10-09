## conda install anaconda::scikit-learn
import pandas as pd
import numpy as np

import torch 
from torch.utils.data import DataLoader

def inverse_dict(data: dict) -> dict:
    ret = {}
    for key, value in data.items():
        ret[value] = key
    return ret

def cal_tSNE(data_loader: DataLoader, label_dict:dict) -> pd.DataFrame:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    full_features = None
    full_labels = None
    for index, (features, labels) in enumerate(data_loader):
        if index == 0:
            full_features = features
            full_labels = labels
        else:
            full_features = torch.cat([full_features, features], dim=0)
            full_labels = torch.cat([full_labels, labels])
    full_features = full_features.detach().cpu().numpy()
    full_labels = full_labels.detach().cpu().numpy()
    
    pca = PCA(n_components=50)
    reduced_data = pca.fit_transform(full_features)

    tsne = TSNE(n_components=2, perplexity=30, method='barnes_hut')
    tsne_features = tsne.fit_transform(reduced_data)

    full_labels = np.array([label_dict[label] for label in full_labels])
    df = pd.DataFrame(columns=['col1', 'col2'], data=tsne_features)
    df['labels'] = full_labels
    return df