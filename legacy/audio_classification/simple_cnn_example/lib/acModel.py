import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

class AudioClassifier(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(5,5), stride=(2,2), padding=(2,2))
        nn.init.kaiming_normal_(self.conv1.weight, a=.1)
        self.conv1.bias.data.zero_()

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        nn.init.kaiming_normal_(self.conv2.weight, a=.1)
        self.conv2.bias.data.zero_()

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        nn.init.kaiming_normal_(self.conv3.weight, a=.1)
        self.conv3.bias.data.zero_()

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        
        
        self.conv = nn.Sequential(
            nn.BatchNorm2d(num_features=2),

            self.conv1,
            nn.ReLU(),
            nn.BatchNorm2d(num_features=8),

            self.conv2,
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16),

            self.conv3,
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32),

            self.conv4,
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64)
        )

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = self.conv(x)

        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        x = self.lin(x)

        return F.softmax(x, dim=1)

class Processor():
        
    @staticmethod
    def training(
        model: nn.Module, train_dl: DataLoader, num_epochs: int, optimizer: optim.Optimizer, loss_fn,
        device='cpu', silent_mod: bool = False
    ):
        # loss_fn = nn.CrossEntropyLoss().to(device=device)
        # optimizer = optim.SGD(model.parameters(), lr=.1, momentum=.9)
        # scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=.1, steps_per_epoch=int(len(train_dl)), 
        #                                       epochs=num_epochs, anneal_strategy='linear')

        for epoch in range(num_epochs):
            ttl_loss = .0
            ttl_correct_pred = .0
            ttl_pred = .0

            for (feature, labels) in train_dl:
                model.train()
                feature, labels = feature.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(feature)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                # scheduler.step()

                ttl_loss += loss
                _, pred = torch.max(outputs, dim=1)
                _, class_id = torch.max(labels, dim=1)
                ttl_correct_pred += (pred == class_id).sum().item()
                ttl_pred += outputs.shape[0]

            batch_num = len(train_dl)
            avg_loss = ttl_loss / batch_num
            accuracy = ttl_correct_pred / ttl_pred * 100
            if not silent_mod: 
                print(f'Epoch: {epoch}, Loss:{avg_loss:.2f}, accuracy:{accuracy:.2f}%')
        if not silent_mod: print('Training End!!')

    @staticmethod
    def simple_training(model: nn.Module, train_dl: DataLoader, num_epochs: int, device='cpu', silent_mod: bool = False):
        loss_fn = nn.CrossEntropyLoss().to(device=device)
        optimizer = optim.Adam(model.parameters(), lr=.001)
        Processor.training(model=model, train_dl=train_dl, num_epochs=num_epochs, device=device, loss_fn=loss_fn, 
            optimizer=optimizer, silent_mod=silent_mod)


    @staticmethod
    def inference(model: nn.Module, val_dl: DataLoader, device='cpu'):
        ttl_correct_pred = .0
        ttl_pred = .0

        model.eval()
        with torch.no_grad():
            for (feature, labels) in val_dl:
                feature, labels = feature.to(device), labels.to(device)
                outputs = model(feature)
                _, pred = torch.max(outputs, dim=1)
                _, class_id = torch.max(labels, dim=1)
                ttl_correct_pred += (pred == class_id).sum().item()
                ttl_pred += outputs.shape[0]
        
        accuracy = ttl_correct_pred / ttl_pred * 100
        print(f'Validation accuracy: {accuracy:2f}%, Validation sample size is: {ttl_pred}')