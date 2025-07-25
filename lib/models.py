import torch.nn as nn

# torchaudio.models.WaveRNN(upsample_scales=[5,5,8], n_classes=10, hop_length=200, kernel_size=5)
# specgram = transforms.MelSpectrogram(sample_rate=sample_rate, hop_length=197, n_fft=400)(waveform)
# outputs = waveRnn(waveform, specgram)

class WavClassifier(nn.Module):
    def __init__(self, class_num=10, l1_in_features=64, c1_in_channels=1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.conv1 = nn.Conv2d(in_channels=c1_in_channels, out_channels=8, kernel_size=(5,5), stride=(2,2), padding=(2,2))
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
            nn.BatchNorm2d(num_features=c1_in_channels),

            self.conv1,
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),

            self.conv2,
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),

            self.conv3,
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            self.conv4,
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=l1_in_features, out_features=class_num)

    def forward(self, x):
        x = self.conv(x)

        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        x = self.lin(x)

        return nn.functional.softmax(x, dim=1)
    
class ElasticRestNet50(nn.Module):
    def __init__(self, class_num: int) -> None:
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        self.body = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.fc = nn.Linear(in_features=self.body.fc.out_features, out_features=class_num)

    def forward(self, x):
        x = self.body(x)
        return self.fc(x)
    
class ElasticRestNet(nn.Module):
    def __init__(self, class_num:int, depth:int) -> None:
        super().__init__()
        assert depth in [34, 50], 'No support'
        from torchvision.models import resnet50, resnet34, ResNet50_Weights, ResNet34_Weights
        if depth == 34:
            self.body = resnet34(weights=ResNet34_Weights.DEFAULT)
        elif depth == 50:
            self.body = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.fc = nn.Linear(in_features=self.body.fc.out_features, out_features=class_num)
    
    def forward(self, x):
        x = self.body(x)
        return self.fc(x)