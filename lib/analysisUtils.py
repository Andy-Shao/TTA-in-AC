import pandas as pd
import os
from enum import Enum

class DatasetType(Enum):
    Audio_MINIST='Audio MNIST',
    Speech_Commands='Speech Commands',
    Speech_Command_Numbers='Speech Command Numbers',
    Speech_Command_Random='Random Speech Commands'

class TTA_Type(Enum):
    TENT='Tent Adaptation'
    NORM='Norm Adaptation'
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
    TTA_Type='TTA type'
    Records='records'
    Model='model'

def load_records(root_pathes:dict[DatasetType, dict[TTA_Type, str]]) -> dict[DatasetType, pd.DataFrame]:
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
        dataset:str, records: pd.DataFrame, tta_type: str, model:str, corruption:str, severity_levels:list[float], 
        tta_operation:str, algorithm:str
    ) -> pd.DataFrame:
    model_weights = pd.DataFrame(columns=['TTA type', 'model', 'number of weight'])
    model_weights.loc[len(model_weights)] = [tta_type, model, search(records, algorithm=algorithm)[RecordColumn.Weight_Num.value].iloc[0]]
    model_weights.insert(loc=3, column='dataset', value=dataset)
    search_value = search(records, algorithm=algorithm, severity_level=0., corruption=pd.NA, tta_operation=pd.NA)[RecordColumn.Accuracy.value]
    model_weights.insert(loc=4, column='origin (%)', value=search_value.iloc[0] if len(search_value)>0 else 'NaN')
    for index, sl in enumerate(severity_levels):
        search_value = search(df=records, algorithm=algorithm, severity_level=sl,corruption=corruption, tta_operation=pd.NA)[RecordColumn.Accuracy.value]
        model_weights.insert(loc=index*2+5, column=f'corrupted ({sl})', value= search_value.iloc[0] if len(search_value)>0 else 'NaN')
        search_value = search(df=records, algorithm=algorithm, severity_level=sl, corruption=corruption, tta_operation=tta_operation)[RecordColumn.Accuracy.value]
        model_weights.insert(loc=index*2+6, column=f'adapted ({sl})', value= search_value.iloc[0] if len(search_value)>0 else 'NaN')
    return model_weights

def analyze_multi_model_accu(configs:dict[str, dict[RecordColumn, object]]) -> pd.DataFrame:
    dataframes = []
    for _, config in configs.items():
        df = analyze_model_accu(
            dataset=config[RecordColumn.Dataset], records=config[RecordColumn.Records], tta_type=config[RecordColumn.TTA_Type], 
            model=config[RecordColumn.Model], corruption=config[RecordColumn.Corruption], 
            severity_levels=config[RecordColumn.Severity_Level], tta_operation=config[RecordColumn.TTA_OP], 
            algorithm=config[RecordColumn.Algorithm])
        dataframes.append(df)
    return pd.concat(dataframes, axis=0, ignore_index=True)

def analyze_guassian_noise(all_records: dict[DatasetType, pd.DataFrame]) -> pd.DataFrame:
    return analyze_multi_model_accu(configs={
        0: {
            RecordColumn.Dataset: DatasetType.Audio_MINIST.value,
            RecordColumn.Records: all_records[DatasetType.Audio_MINIST][TTA_Type.TENT],
            RecordColumn.TTA_Type: TTA_Type.TENT.value,
            RecordColumn.Corruption: 'gaussian_noise',
            RecordColumn.Severity_Level: [0.005],
            RecordColumn.Algorithm: 'restnet50',
            RecordColumn.TTA_OP: 'Tent Adaptation',
            RecordColumn.Model: 'RestNet50'
        },
        1: {
            RecordColumn.Dataset: DatasetType.Audio_MINIST.value,
            RecordColumn.Records: all_records[DatasetType.Audio_MINIST][TTA_Type.TENT],
            RecordColumn.TTA_Type: TTA_Type.NORM.value,
            RecordColumn.Corruption: 'gaussian_noise',
            RecordColumn.Severity_Level: [0.005],
            RecordColumn.Algorithm: 'restnet50',
            RecordColumn.TTA_OP: 'Norm Adaptation',
            RecordColumn.Model: 'RestNet50'
        },
        2: {
            RecordColumn.Dataset: DatasetType.Audio_MINIST.value,
            RecordColumn.Records: all_records[DatasetType.Audio_MINIST][TTA_Type.TTT],
            RecordColumn.TTA_Type: TTA_Type.TTT.value,
            RecordColumn.Corruption: 'gaussian_noise',
            RecordColumn.Severity_Level: [0.005],
            RecordColumn.Algorithm: None,
            RecordColumn.TTA_OP: 'TTT, ts, bn, online',
            RecordColumn.Model: 'Transfer Learning'
        },
        3: {
            RecordColumn.Dataset: DatasetType.Audio_MINIST.value,
            RecordColumn.Records: all_records[DatasetType.Audio_MINIST][TTA_Type.CONMIX],
            RecordColumn.TTA_Type: TTA_Type.CONMIX.value,
            RecordColumn.Corruption: 'gaussian_noise',
            RecordColumn.Severity_Level: [0.005],
            RecordColumn.Algorithm: None,
            RecordColumn.TTA_OP: TTA_Type.CONMIX.value,
            RecordColumn.Model: 'R50+ViT-B_16'
        },
        4: {
            RecordColumn.Dataset: DatasetType.Speech_Commands.value,
            RecordColumn.Records: all_records[DatasetType.Speech_Commands][TTA_Type.TENT],
            RecordColumn.TTA_Type: TTA_Type.TENT.value,
            RecordColumn.Corruption: 'gaussian_noise',
            RecordColumn.Severity_Level: [0.005],
            RecordColumn.Algorithm: 'restnet50',
            RecordColumn.TTA_OP: 'Tent Adaptation + normalized',
            RecordColumn.Model: 'RestNet50'
        },
        5: {
            RecordColumn.Dataset: DatasetType.Speech_Commands.value,
            RecordColumn.Records: all_records[DatasetType.Speech_Commands][TTA_Type.TENT],
            RecordColumn.TTA_Type: TTA_Type.NORM.value,
            RecordColumn.Corruption: 'gaussian_noise',
            RecordColumn.Severity_Level: [0.005],
            RecordColumn.Algorithm: 'restnet50',
            RecordColumn.TTA_OP: 'Norm Adaptation + normalized',
            RecordColumn.Model: 'RestNet50'
        },
        6: {
            RecordColumn.Dataset: DatasetType.Speech_Commands.value,
            RecordColumn.Records: all_records[DatasetType.Speech_Commands][TTA_Type.TTT],
            RecordColumn.TTA_Type: TTA_Type.TTT.value,
            RecordColumn.Corruption: 'gaussian_noise',
            RecordColumn.Severity_Level: [0.005],
            RecordColumn.Algorithm: None,
            RecordColumn.TTA_OP: 'TTT, ts, bn, online',
            RecordColumn.Model: 'Transfer Learning'
        },
        7: {
            RecordColumn.Dataset: DatasetType.Speech_Commands.value,
            RecordColumn.Records: all_records[DatasetType.Speech_Commands][TTA_Type.CONMIX],
            RecordColumn.TTA_Type: TTA_Type.CONMIX.value,
            RecordColumn.Corruption: 'gaussian_noise',
            RecordColumn.Severity_Level: [0.005],
            RecordColumn.Algorithm: None,
            RecordColumn.TTA_OP: 'CoNMix-STDA',
            RecordColumn.Model: 'R50+ViT-B_16'
        },
        8: {
            RecordColumn.Dataset: DatasetType.Speech_Command_Random.value,
            RecordColumn.Records: all_records[DatasetType.Speech_Command_Random][TTA_Type.TENT],
            RecordColumn.TTA_Type: TTA_Type.TENT.value,
            RecordColumn.Corruption: 'gaussian_noise',
            RecordColumn.Severity_Level: [0.005],
            RecordColumn.Algorithm: 'restnet50',
            RecordColumn.TTA_OP: 'Tent Adaptation + normalized',
            RecordColumn.Model: 'RestNet50'
        },
        9: {
            RecordColumn.Dataset: DatasetType.Speech_Command_Random.value,
            RecordColumn.Records: all_records[DatasetType.Speech_Command_Random][TTA_Type.TENT],
            RecordColumn.TTA_Type: TTA_Type.NORM.value,
            RecordColumn.Corruption: 'gaussian_noise',
            RecordColumn.Severity_Level: [0.005],
            RecordColumn.Algorithm: 'restnet50',
            RecordColumn.TTA_OP: 'Norm Adaptation + normalized',
            RecordColumn.Model: 'RestNet50'
        },
        10: {
            RecordColumn.Dataset: DatasetType.Speech_Command_Random.value,
            RecordColumn.Records: all_records[DatasetType.Speech_Command_Random][TTA_Type.TTT],
            RecordColumn.TTA_Type: TTA_Type.TTT.value,
            RecordColumn.Corruption: 'gaussian_noise',
            RecordColumn.Severity_Level: [0.005],
            RecordColumn.Algorithm: None,
            RecordColumn.TTA_OP: 'TTT, ts, bn, online',
            RecordColumn.Model: 'Transfer Learning'
        }
    })

def analyze_background(all_records: dict[DatasetType, pd.DataFrame], noise_type:str) -> pd.DataFrame:
    return analyze_multi_model_accu(configs={
        0: {
            RecordColumn.Dataset: DatasetType.Audio_MINIST.value,
            RecordColumn.Records: all_records[DatasetType.Audio_MINIST][TTA_Type.TENT],
            RecordColumn.TTA_Type: TTA_Type.TENT.value,
            RecordColumn.Corruption: noise_type+'-rand',
            RecordColumn.Severity_Level: [10.0, 3.0],
            RecordColumn.Algorithm: 'restnet50',
            RecordColumn.TTA_OP: 'Tent Adaptation',
            RecordColumn.Model: 'RestNet50'
        },
        1: {
            RecordColumn.Dataset: DatasetType.Audio_MINIST.value,
            RecordColumn.Records: all_records[DatasetType.Audio_MINIST][TTA_Type.TENT],
            RecordColumn.TTA_Type: TTA_Type.NORM.value,
            RecordColumn.Corruption: noise_type+'-rand',
            RecordColumn.Severity_Level: [10.0, 3.0],
            RecordColumn.Algorithm: 'restnet50',
            RecordColumn.TTA_OP: 'Norm Adaptation',
            RecordColumn.Model: 'RestNet50'
        },
        2: {
            RecordColumn.Dataset: DatasetType.Audio_MINIST.value,
            RecordColumn.Records: all_records[DatasetType.Audio_MINIST][TTA_Type.TTT],
            RecordColumn.TTA_Type: TTA_Type.TTT.value,
            RecordColumn.Corruption: noise_type+'-rand',
            RecordColumn.Severity_Level: [10.0, 3.0],
            RecordColumn.Algorithm: None,
            RecordColumn.TTA_OP: 'TTT, ts, bn, online',
            RecordColumn.Model: 'Transfer Learning'
        },
        3: {
            RecordColumn.Dataset: DatasetType.Audio_MINIST.value,
            RecordColumn.Records: all_records[DatasetType.Audio_MINIST][TTA_Type.CONMIX],
            RecordColumn.TTA_Type: TTA_Type.CONMIX.value,
            RecordColumn.Corruption: noise_type,
            RecordColumn.Severity_Level: [10.0, 3.0],
            RecordColumn.Algorithm: None,
            RecordColumn.TTA_OP: TTA_Type.CONMIX.value,
            RecordColumn.Model: 'R50+ViT-B_16'
        },
        4: {
            RecordColumn.Dataset: DatasetType.Speech_Commands.value,
            RecordColumn.Records: all_records[DatasetType.Speech_Commands][TTA_Type.TENT],
            RecordColumn.TTA_Type: TTA_Type.TENT.value,
            RecordColumn.Corruption: noise_type,
            RecordColumn.Severity_Level: [10.0, 3.0],
            RecordColumn.Algorithm: 'restnet50',
            RecordColumn.TTA_OP: 'Tent Adaptation + normalized',
            RecordColumn.Model: 'RestNet50'
        },
        5: {
            RecordColumn.Dataset: DatasetType.Speech_Commands.value,
            RecordColumn.Records: all_records[DatasetType.Speech_Commands][TTA_Type.TENT],
            RecordColumn.TTA_Type: TTA_Type.NORM.value,
            RecordColumn.Corruption: noise_type,
            RecordColumn.Severity_Level: [10.0, 3.0],
            RecordColumn.Algorithm: 'restnet50',
            RecordColumn.TTA_OP: 'Norm Adaptation + normalized',
            RecordColumn.Model: 'RestNet50'
        },
        6: {
            RecordColumn.Dataset: DatasetType.Speech_Commands.value,
            RecordColumn.Records: all_records[DatasetType.Speech_Commands][TTA_Type.TTT],
            RecordColumn.TTA_Type: TTA_Type.TTT.value,
            RecordColumn.Corruption: noise_type,
            RecordColumn.Severity_Level: [10.0, 3.0],
            RecordColumn.Algorithm: None,
            RecordColumn.TTA_OP: 'TTT, ts, bn, online',
            RecordColumn.Model: 'Transfer Learning'
        },
        7: {
            RecordColumn.Dataset: DatasetType.Speech_Commands.value,
            RecordColumn.Records: all_records[DatasetType.Speech_Commands][TTA_Type.CONMIX],
            RecordColumn.TTA_Type: TTA_Type.CONMIX.value,
            RecordColumn.Corruption: noise_type,
            RecordColumn.Severity_Level: [10.0, 3.0],
            RecordColumn.Algorithm: None,
            RecordColumn.TTA_OP: 'CoNMix-STDA',
            RecordColumn.Model: 'R50+ViT-B_16'
        },
        8: {
            RecordColumn.Dataset: DatasetType.Speech_Command_Random.value,
            RecordColumn.Records: all_records[DatasetType.Speech_Command_Random][TTA_Type.TENT],
            RecordColumn.TTA_Type: TTA_Type.TENT.value,
            RecordColumn.Corruption: noise_type,
            RecordColumn.Severity_Level: [10.0, 3.0],
            RecordColumn.Algorithm: 'restnet50',
            RecordColumn.TTA_OP: 'Tent Adaptation + normalized',
            RecordColumn.Model: 'RestNet50'
        },
        9: {
            RecordColumn.Dataset: DatasetType.Speech_Command_Random.value,
            RecordColumn.Records: all_records[DatasetType.Speech_Command_Random][TTA_Type.TENT],
            RecordColumn.TTA_Type: TTA_Type.NORM.value,
            RecordColumn.Corruption: noise_type,
            RecordColumn.Severity_Level: [10.0, 3.0],
            RecordColumn.Algorithm: 'restnet50',
            RecordColumn.TTA_OP: 'Norm Adaptation + normalized',
            RecordColumn.Model: 'RestNet50'
        },
        10: {
            RecordColumn.Dataset: DatasetType.Speech_Command_Random.value,
            RecordColumn.Records: all_records[DatasetType.Speech_Command_Random][TTA_Type.TTT],
            RecordColumn.TTA_Type: TTA_Type.TTT.value,
            RecordColumn.Corruption: noise_type,
            RecordColumn.Severity_Level: [10.0, 3.0],
            RecordColumn.Algorithm: None,
            RecordColumn.TTA_OP: 'TTT, ts, bn, online',
            RecordColumn.Model: 'Transfer Learning'
        }
    })