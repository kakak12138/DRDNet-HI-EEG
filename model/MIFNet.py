import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from model.MamBa.model import SS1D
from torchinfo import summary
from timm.models.layers import trunc_normal_


class Conv(nn.Module):
    def __init__(self, conv, activation=None, bn=None):
        nn.Module.__init__(self)
        self.conv = conv
        self.activation = activation
        if bn:
            self.conv.bias = None
        self.bn = bn

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class VarPoold(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        t = x.shape[2]
        out_shape = (t - self.kernel_size) // self.stride + 1
        out = []
        for i in range(out_shape):
            index = i * self.stride
            input = x[:, :, index:index + self.kernel_size]
            output = torch.log(torch.clamp(input.var(dim=-1, keepdim=True), 1e-6, 1e6))
            out.append(output)
        out = torch.cat(out, dim=-1)

        return out


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=0.5, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


class MIFNet(nn.Module):
    def __init__(self, num_classes, chans=32, out_planes=64, embded_dim=64, pool_size=50, pool_stride=15,
                 radix=2):
        super().__init__()
        self.in_planes = chans * radix
        self.mid_planes = out_planes * radix
        self.out_planes = out_planes
        self.sconv = Conv(nn.Conv1d(self.in_planes, self.mid_planes, 1, bias=False, groups=radix),
                          bn=nn.BatchNorm1d(self.mid_planes), activation=None)
        self.TMamba = SS1D(64,dropout=0.5)
        self.Time_avg = nn.AvgPool1d(pool_size, pool_stride)
        self.var_pool = VarPoold(pool_size, pool_stride)
        self.max_pool=nn.MaxPool1d(pool_size,pool_stride)
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(self.out_planes, self.out_planes, (2, 1)),
            nn.BatchNorm2d(self.out_planes),
        )
        self.dropout = nn.Dropout(0.5)
        # self.classify = LinearWithConstraint(out_planes * embded_dim, num_classes, doWeightNorm=True)
        self.classify = LinearWithConstraint(4096, num_classes, doWeightNorm=True)
        self.apply(self.initParms)

    def initParms(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.squeeze(dim=1)  # B(N C T)
        x = self.sconv(x)
        x = torch.split(x, self.out_planes, dim=1)
        x1 = x[0].unsqueeze(dim=2)
        x2 = x[1].unsqueeze(dim=2)
        x = torch.cat((x1, x2), dim=2)
        x = self.conv_encoder(x)
        x = x.squeeze(dim=2)
        x = x.transpose(1, 2)  # N T C
        x = self.TMamba(x)
        x = x.transpose(1, 2)  # N T C
        x = self.Time_avg(x)
        x = self.dropout(x)
        x = self.classify(x.flatten(1))
        return x

if __name__ == "__main__":
    input_size = (9, 1, 64, 1000)
    model = MIFNet(6).to("cuda")
    summary(model, input_size)