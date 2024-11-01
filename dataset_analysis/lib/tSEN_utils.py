## conda install anaconda::scikit-learn==1.5.1
import pandas as pd
import numpy as np

import torch 
from torch.utils.data import DataLoader

def inverse_dict(data: dict) -> dict:
    ret = {}
    for key, value in data.items():
        ret[value] = key
    return ret

def cal_tSNE(data_loader: DataLoader, label_dict:dict, reduceable=True) -> pd.DataFrame:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    full_features = None
    full_labels = None
    for index, (features, labels) in enumerate(data_loader):
        batch_size = features.shape[0]
        if index == 0:
            full_features = features.reshape(batch_size, -1)
            full_labels = labels.reshape(-1)
        else:
            full_features = torch.cat([full_features, features.reshape(batch_size, -1)], dim=0)
            full_labels = torch.cat([full_labels, labels.reshape(-1)])
    full_features = full_features.detach().cpu().numpy()
    full_labels = full_labels.detach().cpu().numpy()
    
    if reduceable:
        print('reduced mode')
        pca = PCA(n_components=50)
        reduced_data = pca.fit_transform(full_features)
    else:
        print('unreduced mode')
        reduced_data = full_features

    print(f'calculation features shape: {full_features.shape}')

    tsne = TSNE(n_components=2, perplexity=30, method='barnes_hut')
    tsne_features = tsne.fit_transform(reduced_data)

    full_labels = np.array([label_dict[label] for label in full_labels])
    df = pd.DataFrame(columns=['col1', 'col2'], data=tsne_features)
    df['label'] = full_labels
    return df

def cal_tSNEs(loaders: dict[str, DataLoader], label_dict: dict, reduceable=True) -> pd.DataFrame:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    full_features = None
    full_labels = None
    full_source = None
    flag = False
    for source, loader in loaders.items():
        for index, (features, labels) in enumerate(loader):
            batch_size = features.shape[0]
            if flag == False and index == 0:
                full_features = features.reshape(batch_size, -1)
                full_labels = labels.reshape(-1)
                full_source = np.array([source]).repeat(batch_size)
                flag = True
            else:
                full_features = torch.cat([full_features, features.reshape(batch_size, -1)], dim=0)
                full_labels = torch.cat([full_labels, labels.reshape(-1)])
                full_source = np.concatenate((full_source, np.array([source]).repeat(batch_size)))
    full_features = full_features.detach().cpu().numpy()
    full_labels = full_labels.detach().cpu().numpy()

    print('Finish loading!')

    if reduceable:
        print('reduced mode')
        pca = PCA(n_components=50)
        reduced_data = pca.fit_transform(full_features)
    else:
        print('unreduced mode')
        reduced_data = full_features

    print(f'calculate features shape: {full_features.shape}')

    tsne = TSNE(n_components=2, perplexity=30, method='barnes_hut')
    tsne_features = tsne.fit_transform(reduced_data)

    full_labels = np.array([label_dict[label] for label in full_labels])
    df = pd.DataFrame(columns=['col1', 'col2'], data=tsne_features)
    df['label'] = full_labels
    df['source'] = full_source
    return df

def merg_tSNE(df: pd.DataFrame, mode='mean') -> pd.DataFrame:
    assert mode in ['mean', 'sum'], 'No support'
    ret = pd.DataFrame(columns=['col1', 'col2', 'label'])
    for label in df['label'].unique():
        sub_df = df[df['label'] == label][['col1', 'col2']]
        if mode == 'mean':
            merged_df = sub_df.mean(axis=0)
        elif mode == 'sum':
            merged_df = sub_df.sum(axis=0)
        merged_df['label'] = label
        ret.loc[len(ret)] = merged_df
    return ret

def clip_df(df: pd.DataFrame, min_distance:float, max_distance:float) -> pd.DataFrame:
    return df[(df['distance'] >= min_distance) & (df['distance'] <= max_distance)]

def omit_outline(df: pd.DataFrame) -> pd.DataFrame:
    import math

    ret = pd.DataFrame(columns=['col1', 'col2', 'label'])
    df = df.copy()
    distances = []
    for index, row in df.iterrows():
        distances.append(math.sqrt(row['col1']**2 + row['col2']**2))
    df['distance'] = distances
    for label in df['label'].unique():
        sub_df = df[df['label'] == label]['distance']
        sub_mean = sub_df.mean(axis=0)
        sub_std = sub_df.std(axis=0)
        min_dist = sub_mean - 2 * sub_std
        max_dist = sub_mean + 2 * sub_std
        tmp = clip_df(
            df=df[df['label'] == label], min_distance=min_dist, max_distance=max_dist
        )[['col1', 'col2', 'label']]
        ret = pd.concat([ret, tmp], axis=0, ignore_index=True, copy=True)
    return ret

def sampling(df: pd.DataFrame, max_size:int) -> pd.DataFrame:
    ret = pd.DataFrame(columns=['col1', 'col2', 'label'])
    for label in df['label'].unique():
        sub_df = df[df['label'] == label].sample(n=max_size, replace=False)
        ret = pd.concat([ret, sub_df], axis=0, ignore_index=True, copy=True)
    return ret

def show_tSNE(title:str, train_df: pd.DataFrame=None, test_df: pd.DataFrame=None, point_size=5) -> None:
    import matplotlib.pyplot as plt

    if train_df is not None:
        plt.scatter(train_df['col1'], train_df['col2'], color='green', label='train', s=point_size)
        merged_df = merg_tSNE(df=train_df, mode='mean')
        for index, row in merged_df.iterrows():
            plt.text(row['col1'], row['col2'], row['label'], fontsize=12, ha='right', color='green')

    if test_df is not None:
        plt.scatter(test_df['col1'], test_df['col2'], color='red', label='test', s=point_size)
        merged_df = merg_tSNE(df=test_df, mode='mean')
        for index, row in merged_df.iterrows():
            plt.text(row['col1'], row['col2'], row['label'], fontsize=12, ha='right', color='red')

    plt.legend()
    plt.xlabel('col1')
    plt.ylabel('col2')
    plt.title(title)
    plt.show()

def show_by_label(df: pd.DataFrame, figheight=12, figwidth=12) -> None:
    import matplotlib.pyplot as plt

    def cal_plots(label_num:int) -> tuple[int, int]:
        import math
        heigh = int(math.sqrt(label_num))
        width = heigh
        if heigh * width < label_num:
            width += 1
        return width, heigh

    label_type = df['label'].unique()
    width, heigh = cal_plots(label_num=len(label_type))
    f, axis = plt.subplots(width, heigh)
    f.set_figheight(figheight)
    f.set_figwidth(figwidth)

    for index, label in enumerate(label_type):
        data =  df[df['label'] == label]
        data = data.sample(n=800)
        w = index // heigh
        h = index - (w * heigh)
        axis[w, h].scatter(data['col1'], data['col2'], color='lightblue', label=label)
        axis[w, h].set_title(label)
        axis[w, h].axis('off')
    if len(label_type) < (width * heigh):
        for index in range(len(label_type), (width * heigh)):
            w = index // heigh
            h = index - (w * heigh)
            axis[w, h].axis('off')
    plt.show()