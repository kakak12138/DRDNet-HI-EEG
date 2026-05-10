import torch
import torch.nn as nn
from torchinfo import summary


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=0.25, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


class EEGNet(nn.Module):
    def __init__(self, n_classes):
        super(EEGNet, self).__init__()

        linear_size = (1000 // (16 * 8)) * 16

        self.conv1 = nn.Conv2d(1, 8, (1, 25), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = Conv2dWithConstraint(8, 16, (32, 1), groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 16))
        self.dropout1 = nn.Dropout(0.5)

        self.conv3 = nn.Conv2d(16, 16, (1, 16), groups=16, bias=False, padding='same')
        self.conv4 = nn.Conv2d(16, 16, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(0.5)

        self.flatten = nn.Flatten()
        self.fc = LinearWithConstraint(linear_size, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    input_size = (9, 1, 32, 1000)
    model = EEGNet(6).to("cuda")
    summary(model, input_size)