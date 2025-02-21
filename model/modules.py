import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.xavier_normal_(m.weight, gain=1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.xavier_normal_(m.weight, gain=1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ModuleList):
            for sub_module in m:
                weight_init(sub_module)
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity, nn.Dropout2d, nn.LeakyReLU, nn.Dropout)):
            pass
        else:
            m.initialize()


class CropLayer(nn.Module):
    # E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]


class asyConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(asyConv, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, kernel_size), stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
            self.initialize()
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)
            self.square_bn = nn.BatchNorm2d(num_features=out_channels)

            center_offset_from_origin_border = padding - kernel_size // 2
            ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border)
            hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + 1)
            if center_offset_from_origin_border >= 0:
                self.ver_conv_crop_layer = nn.Identity()
                ver_conv_padding = ver_pad_or_crop
                self.hor_conv_crop_layer = nn.Identity()
                hor_conv_padding = hor_pad_or_crop
            else:
                self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
                ver_conv_padding = (0, 0)
                self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
                hor_conv_padding = (0, 0)
            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
                                      stride=stride,
                                      padding=ver_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                      stride=stride,
                                      padding=hor_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels)
            self.initialize()

    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            square_outputs = self.square_bn(square_outputs)
            vertical_outputs = self.ver_conv_crop_layer(input)
            vertical_outputs = self.ver_conv(vertical_outputs)
            vertical_outputs = self.ver_bn(vertical_outputs)
            horizontal_outputs = self.hor_conv_crop_layer(input)
            horizontal_outputs = self.hor_conv(horizontal_outputs)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            return square_outputs + vertical_outputs + horizontal_outputs

    def initialize(self):
        weight_init(self)


class RGLA(nn.Module):
    def __init__(self, in_channels, mid_channels=32, out_channels=64):
        super(RGLA, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.local_asyConv = asyConv(out_channels, out_channels, kernel_size=3, padding=1)
        self.local_conv2 = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.conv1x1_q = nn.Conv2d(in_channels, mid_channels, 1)
        self.conv1x1_k = nn.Conv2d(in_channels, mid_channels, 1)
        self.conv1x1_v = nn.Conv2d(in_channels, mid_channels, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1x1_att = nn.Conv2d(mid_channels, out_channels, 1)

        self.bn = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)

        self.initialize()

    def forward(self, x):
        x0 = self.conv0(x)

        # local_attention
        xl = self.local_asyConv(x0)
        local_att = self.sigmoid(self.local_conv2(xl)) * x0

        # global_attention
        query = self.conv1x1_q(x)
        key = self.conv1x1_k(x)
        key = self.max_pool(key)
        value = self.conv1x1_v(x)
        value = self.max_pool(value)
        b, c, h, w = query.shape
        query = query.view(b, c, -1).permute(0, 2, 1)
        key = key.view(b, c, -1)
        query_key = torch.bmm(query, key)
        query_key = F.softmax(query_key, dim=-1)
        value = value.view(b, c, -1).permute(0, 2, 1)
        global_att = torch.bmm(query_key, value).view(b, c, h, w)
        global_att = self.conv1x1_att(global_att)

        fused_feature_maps = local_att + global_att + x0
        return self.dropout(self.leakyrelu(self.bn(fused_feature_maps)))

    def initialize(self):
        weight_init(self)
