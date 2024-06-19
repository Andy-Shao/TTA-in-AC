import torch

def rotate_batch(batch: torch.Tensor, label: str) -> tuple[torch.Tensor, torch.Tensor]:
    if label == 'rand':
        labels = torch.randint(high=4, size=(len(batch),), dtype=torch.long)
    elif label == 'expand':
        labels = torch.cat(tensors=[
            torch.zeros(len(batch), dtype=torch.long),
            torch.zeros(len(batch), dtype=torch.long) + 1,
            torch.zeros(len(batch), dtype=torch.long) + 2,
            torch.zeros(len(batch), dtype=torch.long) + 3
        ])
        batch = batch.repeat((4,1,1,1))
    else:
        assert isinstance(label, int)
        labels = torch.zeros(size=(len(batch),), dtype=torch.long) + label
    return rotate_batch_with_labels(batch=batch, labels=labels), labels

def rotate_batch_with_labels(batch: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    images = []
    for img, label in zip(batch, labels):
        if label == 1:
            img = tensor_rot_90(x=img)
        elif label == 2:
            img = tensor_rot_180(x=img)
        elif label == 3:
            img = tensor_rot_270(x=img)
        images.append(img.unsqueeze(0))
    return torch.cat(images)

# Assumes that tensor is (nchannels, height, width)
def tensor_rot_90(x: torch.Tensor) -> torch.Tensor:
    return x.flip(2).transpose(1, 2)

def tensor_rot_180(x: torch.Tensor) -> torch.Tensor:
    return x.flip(2).flip(1)

def tensor_rot_270(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).flip(2)