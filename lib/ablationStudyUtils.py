import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def format_loss(
        origin:pd.DataFrame, updated:pd.DataFrame, no_pl:pd.DataFrame, no_cst:pd.DataFrame, max_length:int=20
) -> dict[str, dict[str, object]]:
    def format_data(data:pd.DataFrame, ls:str, c:str) -> dict[str, object]:
        return {
            'data': data,
            'linestyle': ls,
            'color': c
        }
    return {
        'org': format_data(data=origin.loc[:max_length], ls='-', c='b'),
        'upd': format_data(data=updated.loc[:max_length], ls='-', c='r'),
        'no_pl': format_data(data=no_pl.loc[:max_length], ls='--', c='b'),
        'no_cst': format_data(data=no_cst.loc[:max_length], ls='-.', c='b')
    }

def loss_ablation(
        x:np.ndarray, 
        title:str,
        lines:dict[str, dict[str, object]],
        xlabel='Epoch',
        ylabel='Accuracy (%)'
    ) -> None:
    for label, line in lines.items():
        plt.plot(x, line['data'].to_numpy(), label=label, ls=line['linestyle'], color=line['color'])
    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)
    plt.title(title)
    plt.legend()
    plt.show()