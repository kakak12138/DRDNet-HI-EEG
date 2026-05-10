import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class ShallowConvNet(nn.Module):
    def __init__(self, n_classes):

        super(ShallowConvNet, self).__init__()

        self.classes_num = n_classes
        in_channels = 32
        time_step = 1000
        self.batch_norm = True
        self.batch_norm_alpha = 0.1
        n_ch1 = 40

        if self.batch_norm:
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 25), stride=1),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(in_channels, 1), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch1, momentum=self.batch_norm_alpha, affine=True, eps=1e-5)
            )
        else:
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 25), stride=1),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(in_channels, 1), stride=1)
            )

        # 自动计算全连接层输入维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, in_channels, time_step)
            out = self.layer1(dummy_input)
            out = torch.square(out)
            out = F.avg_pool2d(out, (1, 75), 15)
            self.n_outputs = out.numel() // out.shape[0]

        self.clf = nn.Linear(self.n_outputs, self.classes_num)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.square(x)
        x = F.avg_pool2d(x, (1, 75), 15)
        x = torch.log(x + 1e-8)  # 避免log(0)
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.flatten(1)
        x = self.clf(x)
        return x


class DeepConvNet(nn.Module):
    def __init__(self, n_classes):

        super(DeepConvNet, self).__init__()

        self.classes_num = n_classes
        in_channels = 32
        time_step = 1000
        self.batch_norm = True
        self.batch_norm_alpha = 0.1
        kernal = 10
        n_ch1 = 25
        n_ch2 = 50
        n_ch3 = 100
        self.n_ch4 = 200

        if self.batch_norm:
            self.convnet = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, kernal), stride=1, bias=False),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(in_channels, 1), stride=1, bias=False),
                nn.BatchNorm2d(n_ch1, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),

                nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, kernal), stride=1, bias=False),
                nn.BatchNorm2d(n_ch2, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),

                nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, kernal), stride=1, bias=False),
                nn.BatchNorm2d(n_ch3, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),

                nn.Conv2d(n_ch3, self.n_ch4, kernel_size=(1, kernal), stride=1, bias=False),
                nn.BatchNorm2d(self.n_ch4, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            )
        else:
            self.convnet = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, kernal), stride=1, bias=False),
                nn.BatchNorm2d(n_ch1, momentum=self.batch_norm_alpha, affine=True, eps=1e-5),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(in_channels, 1), stride=1),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, kernal), stride=1),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, kernal), stride=1),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch3, self.n_ch4, kernel_size=(1, kernal), stride=1),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            )

        # 自动计算全连接层输入维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, in_channels, time_step)
            out = self.convnet(dummy_input)
            self.n_outputs = out.numel() // out.shape[0]

        self.clf = nn.Sequential(
            nn.Linear(self.n_outputs, self.classes_num),
            nn.Dropout(p=0.2)
        )

    def forward(self, x):
        output = self.convnet(x)
        output = output.flatten(1)
        output = self.clf(output)
        return output


if __name__ == "__main__":
    input_size = (9, 1, 32, 1000)
    model = ShallowConvNet(6).to("cuda")
    summary(model, input_size)