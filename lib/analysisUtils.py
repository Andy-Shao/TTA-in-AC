import pandas as pd
import os
from enum import Enum

class TTA_Type(Enum):
    TENT='tent adapt'
    NORM='norm adapt'
    TTT='TTT'
    CONMIX='CoNMix'

class RecordColumn(Enum):
    Dataset='dataset'
    Algorithm='algorithm'
    TTA_OP='tta-operation'
    Corruption='corruption'
    Accuracy='accuracy'
    Error='error'
    Severity_Level='severity level'
    Weight_Num='number of weight'

def load_records(root_pathes:dict[str, dict[TTA_Type, str]]) -> dict[str, pd.DataFrame]:
    result = {}
    for dataset, config in root_pathes.items():
        tmp = {}
        for tta_type, root_path in config.items():
            tmp[tta_type] = combine_csvs(root_path=root_path)
        result[dataset] = tmp
    return result

def combine_csvs(root_path: str) -> pd.DataFrame:
    records = []
    for sub_path in os.listdir(root_path):
        if 'accuracy_record' in sub_path and sub_path.endswith('.csv'):
            record = pd.read_csv(os.path.join(root_path, sub_path), index_col=0)
            records.append(record)
    return pd.concat(records, axis=0, ignore_index=True)

def search(df: pd.DataFrame, dataset=None, algorithm=None, tta_operation=None, corruption=None, severity_level=None):
    if dataset is not None:
        if dataset is not pd.NA:
            df = df[df[RecordColumn.Dataset.value] == dataset]
        else:
            df = df[df[RecordColumn.Dataset.value].isna()]
    if algorithm is not None:
        if algorithm is not pd.NA:
            df = df[df[RecordColumn.Algorithm.value] == algorithm]
        else:
            df = df[df[RecordColumn.Algorithm.value].isna()]
    if tta_operation is not None:
        if tta_operation is not pd.NA:
            df = df[df[RecordColumn.TTA_OP.value] == tta_operation]
        else:
            df = df[df[RecordColumn.TTA_OP.value].isna()]
    if corruption is not None:
        if corruption is not pd.NA:
            df = df[df[RecordColumn.Corruption.value] == corruption]
        else:
            df = df[df[RecordColumn.Corruption.value].isna()]
    if severity_level is not None:
        if severity_level is not pd.NA:
            df = df[df[RecordColumn.Severity_Level.value] == severity_level]
        else:
            df = df[df[RecordColumn.Severity_Level.value].isna()]
    return df

def analyze_model_accu(
        dataset:str, records: pd.DataFram, tta_type: str, model:str, corruption:str, severity_levels:list[float]
    ) -> pd.DataFrame:
    model_weights = pd.DataFrame(columns=['TTA type', 'model', 'number of weight'])
    model_weights.loc[len(model_weights)] = [tta_type, model, search(records, algorithm=model)[RecordColumn.Weight_Num.value].iloc[0]]
    model_weights.insert(loc=3, column='dataset', value=[dataset])
    model_weights.insert(loc=4, column='origin (%)', value=[search(records, algorithm=model, severity_level=0., corruption=pd.NA, tta_operation=pd.NA)[RecordColumn.Accuracy.value].iloc[0]])
    for index, sl in enumerate(severity_levels):
        model_weights.insert(loc=index + 5, column=f'corrupted ({sl})')
    # TODO
    return model_weights