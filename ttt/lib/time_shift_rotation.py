import torch 

from lib.toolkit import BatchTransform
from ttt.lib.prepare_dataset import TimeShiftOps

def rotate_batch(batch: torch.Tensor, label: str, data_transforms: dict[str, BatchTransform]) -> tuple[torch.Tensor, torch.Tensor]:
    if label == 'rand':
        labels = torch.randint(low=0, high=3, size=(len(batch),), dtype=torch.long)
    elif label == 'expand':
        labels = torch.cat(tensors=[
            torch.zeros(len(batch), dtype=torch.long),
            torch.zeros(len(batch), dtype=torch.long) + 1,
            torch.zeros(len(batch), dtype=torch.long) + 2
        ])
        batch = batch.repeat((3, 1, 1, 1))
    else:
        raise Exception('No support')
    return rotate_batch_with_labels(batch=batch, labels=labels, data_transforms=data_transforms), labels

def rotate_batch_with_labels(batch: torch.Tensor, labels: torch.Tensor, data_transforms: dict[str, BatchTransform]) -> torch.Tensor:
    audios = []
    for audio, label in zip(batch, labels):
        if label == 1: # left time shift
            audio = data_transforms[TimeShiftOps.LEFT].tran_one(audio)
        elif label == 2: # right time shift
            audio = data_transforms[TimeShiftOps.RIGHT].tran_one(audio)
        else: # without the time shift
            audio = data_transforms[TimeShiftOps.ORIGIN].tran_one(audio)
        audios.append(audio)
    return torch.cat(audios)