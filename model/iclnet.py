import torch
from torch.utils import model_zoo
from torchvision.models.resnet import resnet50
from model.swin_encoder import SwinTransformer
from model.resnet_encoder import ResNet
from model.vgg_encoder import VGG
from model.pvtv2_encoder import pvt_v2_b4
from model.cyclemlp_encoder import CycleMLP_B4
from model.caps_decoder import CapsDecoder
from model.modules import RGLA
# from pytorch_grad_cam import GradCAM, ScoreCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image
import torch.nn as nn
import torch.nn.functional as F


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.xavier_normal_(m.weight, gain=1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
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
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()


class ICLNet(torch.nn.Module):
    def __init__(self, cfg, model_name='ICLNet-R'):
        super(ICLNet, self).__init__()
        self.cfg = cfg
        self.model_name = model_name

        if self.model_name == 'ICLNet-R':
            # ResNet Encoder
            self.encoder = ResNet(self.cfg)
            pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
            encoder_dict = self.encoder.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
            encoder_dict.update(pretrained_dict)
            self.encoder.load_state_dict(encoder_dict)

            self.rgla2 = RGLA(256)
            self.rgla3 = RGLA(512)
            self.rgla4 = RGLA(1024)

            self.decoder = CapsDecoder(in_channels=2048)

        elif self.model_name == 'ICLNet-S':
            # Swin Encoder
            self.encoder = SwinTransformer(
                img_size=384,
                embed_dim=128,
                depths=[2, 2, 18, 2],
                num_heads=[4, 8, 16, 32],
                window_size=12
            )
            pretrained_dict = torch.load('checkpoint/Backbone/Swin/swin_base_patch4_window12_384_22k.pth')["model"]
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.encoder.state_dict()}
            self.encoder.load_state_dict(pretrained_dict)

            self.rgla1 = RGLA(128)
            self.rgla2 = RGLA(256)
            self.rgla3 = RGLA(512)

            self.decoder = CapsDecoder(in_channels=1024)

        elif self.model_name == 'ICLNet-P':
            # PVT Encoder
            self.encoder = pvt_v2_b4()

            pretrained_dict = torch.load('checkpoint/Backbone/PVTv2/pvt_v2_b4.pth')
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.encoder.state_dict()}
            self.encoder.load_state_dict(pretrained_dict)

            self.rgla2 = RGLA(64)
            self.rgla3 = RGLA(128)
            self.rgla4 = RGLA(320)

            self.decoder = CapsDecoder(in_channels=512)

        elif self.model_name == "ICLNet-V":
            # VGG Encoder
            self.encoder = VGG()

            self.rgla2 = RGLA(128)
            self.rgla3 = RGLA(256)
            self.rgla4 = RGLA(512)

            self.decoder = CapsDecoder(in_channels=512)

        elif self.model_name == "ICLNet-M":
            # MLP Encoder
            self.encoder = CycleMLP_B4()

            pretrained_dict = torch.load('checkpoint/Backbone/CycleMLP/CycleMLP_B4.pth')
            pretrained_dict = pretrained_dict['model']
            encoder_dict = self.encoder.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
            self.encoder.load_state_dict(pretrained_dict, strict=False)

            self.rgla2 = RGLA(64)
            self.rgla3 = RGLA(128)
            self.rgla4 = RGLA(320)

            self.decoder = CapsDecoder(in_channels=512)

        else:
            print("UNDEFINED BACKBONE NAME.")

        self.initialize()

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x, shape=None, name=None):
        features = self.encoder(x)

        if self.model_name == 'ICLNet-R':
            x1 = features[0]
            x2 = features[1]
            x3 = features[2]
            x4 = features[3]
            x5 = features[4]
            x2 = self.rgla2(x2)
            x3 = self.rgla3(x3)
            x4 = self.rgla4(x4)
            out5, out4, out3, out2 = self.decoder(x2, x3, x4, x5)

        elif self.model_name == 'ICLNet-S':
            x1 = features[0]
            x2 = features[1]
            x3 = features[2]
            x4 = features[3]
            x5 = features[4]
            x1 = self.rgla1(x1)
            x2 = self.rgla2(x2)
            x3 = self.rgla3(x3)
            out5, out4, out3, out2 = self.decoder(x1, x2, x3, x4)

        elif self.model_name == 'ICLNet-P':
            x1 = features[0]
            x2 = features[1]
            x3 = features[2]
            x4 = features[3]
            x5 = features[4]
            x1 = self.rgla1(x1)
            x2 = self.rgla2(x2)
            x3 = self.rgla3(x3)
            out5, out4, out3, out2 = self.decoder(x1, x2, x3, x4)

        elif self.model_name == 'ICLNet-V':
            x1 = features[0]
            x2 = features[1]
            x3 = features[2]
            x4 = features[3]
            x5 = features[4]
            x2 = self.rgla2(x2)
            x3 = self.rgla3(x3)
            x4 = self.rgla4(x4)
            out5, out4, out3, out2 = self.decoder(x2, x3, x4, x5)

        elif self.model_name == 'ICLNet-M':
            x1 = features[0]
            x2 = features[1]
            x3 = features[2]
            x4 = features[3]
            x5 = features[4]
            x1 = self.rgla1(x1)
            x2 = self.rgla2(x2)
            x3 = self.rgla3(x3)
            out5, out4, out3, out2 = self.decoder(x1, x2, x3, x4)

        if shape is None:
            shape = x.size()[2:]

        pred5 = F.interpolate(out5, size=shape, mode='bilinear', align_corners=True)
        pred4 = F.interpolate(out4, size=shape, mode='bilinear', align_corners=True)
        pred3 = F.interpolate(out3, size=shape, mode='bilinear', align_corners=True)
        pred2 = F.interpolate(out2, size=shape, mode='bilinear', align_corners=True)

        return pred5, pred4, pred3, pred2

    def initialize(self):
        if self.cfg.checkpoint:
            self.load_state_dict(torch.load(self.cfg.checkpoint))
        else:
            weight_init(self)
