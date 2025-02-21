import torch
import torch.nn as nn
import torch.nn.functional as F
import math


eps = 1e-12


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
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity, nn.Dropout2d, nn.LeakyReLU)):
            pass
        else:
            m.initialize()


class EmRouting2d(nn.Module):
    def __init__(self, A, B, caps_size, kernel_size=3, stride=1, padding=1, iters=3, final_lambda=1e-2):
        super(EmRouting2d, self).__init__()
        self.A = A
        self.B = B
        self.psize = caps_size
        self.mat_dim = int(caps_size ** 0.5)

        self.k = kernel_size
        self.kk = kernel_size ** 2
        self.kkA = self.kk * A

        self.stride = stride
        self.pad = padding

        self.iters = iters

        self.W = nn.Parameter(torch.FloatTensor(self.kkA, B, self.mat_dim, self.mat_dim))
        nn.init.kaiming_uniform_(self.W.data)

        self.beta_u = nn.Parameter(torch.FloatTensor(1, 1, B, 1))
        self.beta_a = nn.Parameter(torch.FloatTensor(1, 1, B))
        nn.init.constant_(self.beta_u, 0)
        nn.init.constant_(self.beta_a, 0)

        self.final_lambda = final_lambda
        self.ln_2pi = math.log(2*math.pi)
        self.initialize()

    def m_step(self, v, a_in, r):
        # v: [b, l, kkA, B, psize]
        # a_in: [b, l, kkA]
        # r: [b, l, kkA, B, 1]
        b, l, _, _, _ = v.shape

        # r: [b, l, kkA, B, 1]
        r = r * a_in.view(b, l, -1, 1, 1)
        # r_sum: [b, l, 1, B, 1]
        r_sum = r.sum(dim=2, keepdim=True)
        # coeff: [b, l, kkA, B, 1]
        coeff = r / (r_sum + eps)

        # mu: [b, l, 1, B, psize]
        mu = torch.sum(coeff * v, dim=2, keepdim=True)
        # sigma_sq: [b, l, 1, B, psize]
        sigma_sq = torch.sum(coeff * (v - mu)**2, dim=2, keepdim=True) + eps

        # [b, l, B, 1]
        r_sum = r_sum.squeeze(2)
        # [b, l, B, psize]
        sigma_sq = sigma_sq.squeeze(2)
        # [1, 1, B, 1] + [b, l, B, psize] * [b, l, B, 1]
        cost_h = (self.beta_u + torch.log(sigma_sq.sqrt())) * r_sum
        # cost_h = (torch.log(sigma_sq.sqrt())) * r_sum

        # [b, l, B]
        a_out = torch.sigmoid(self.lambda_*(self.beta_a - cost_h.sum(dim=3)))
        # a_out = torch.sigmoid(self.lambda_*(-cost_h.sum(dim=3)))

        return a_out, mu, sigma_sq

    def e_step(self, v, a_out, mu, sigma_sq):
        b, l, _ = a_out.shape
        # v: [b, l, kkA, B, psize]
        # a_out: [b, l, B]
        # mu: [b, l, 1, B, psize]
        # sigma_sq: [b, l, B, psize]

        # [b, l, 1, B, psize]
        sigma_sq = sigma_sq.unsqueeze(2)

        ln_p_j = -0.5 * torch.sum(torch.log(sigma_sq*self.ln_2pi), dim=-1) \
            - torch.sum((v - mu)**2 / (2 * sigma_sq), dim=-1)

        # [b, l, kkA, B]
        ln_ap = ln_p_j + torch.log(a_out.view(b, l, 1, self.B))
        # [b, l, kkA, B]
        r = torch.softmax(ln_ap, dim=-1)
        # [b, l, kkA, B, 1]
        return r.unsqueeze(-1)

    def forward(self, a_in, pose):
        # pose: [batch_size, A, psize]
        # a: [batch_size, A]
        batch_size = a_in.shape[0]

        # a: [b, A, h, w]
        # pose: [b, A*psize, h, w]
        b, _, h, w = a_in.shape

        # [b, A*psize*kk, l]
        pose = F.unfold(pose, self.k, stride=self.stride, padding=self.pad)
        l = pose.shape[-1]
        # [b, A, psize, kk, l]
        pose = pose.view(b, self.A, self.psize, self.kk, l)
        # [b, l, kk, A, psize]
        pose = pose.permute(0, 4, 3, 1, 2).contiguous()
        # [b, l, kkA, psize]
        pose = pose.view(b, l, self.kkA, self.psize)
        # [b, l, kkA, 1, mat_dim, mat_dim]
        pose = pose.view(batch_size, l, self.kkA, self.mat_dim, self.mat_dim).unsqueeze(3)

        # [b, l, kkA, B, mat_dim, mat_dim]
        pose_out = torch.matmul(pose, self.W)

        # [b, l, kkA, B, psize]
        v = pose_out.view(batch_size, l, self.kkA, self.B, -1)

        # [b, kkA, l]
        a_in = F.unfold(a_in, self.k, stride=self.stride, padding=self.pad)
        # [b, A, kk, l]
        a_in = a_in.view(b, self.A, self.kk, l)
        # [b, l, kk, A]
        a_in = a_in.permute(0, 3, 2, 1).contiguous()
        # [b, l, kkA]
        a_in = a_in.view(b, l, self.kkA)

        r = a_in.new_ones(batch_size, l, self.kkA, self.B, 1)
        for i in range(self.iters):
            # this is from open review
            self.lambda_ = self.final_lambda * (1 - 0.95 ** (i+1))
            a_out, pose_out, sigma_sq = self.m_step(v, a_in, r)
            if i < self.iters - 1:
                r = self.e_step(v, a_out, pose_out, sigma_sq)

        # [b, l, B*psize]
        pose_out = pose_out.squeeze(2).view(b, l, -1)
        # [b, B*psize, l]
        pose_out = pose_out.transpose(1, 2)
        # [b, B, l]
        a_out = a_out.transpose(1, 2).contiguous()

        oh = ow = math.floor(l**(1/2))

        a_out = a_out.view(b, -1, oh, ow)
        pose_out = pose_out.view(b, -1, oh, ow)

        return a_out, pose_out

    def initialize(self):
        weight_init(self)


class CapsDecoder(nn.Module):
    def __init__(self, in_channels):
        super(CapsDecoder, self).__init__()
        self.num_caps = 8
        planes = 16

        self.conv_m = nn.Conv2d(in_channels, self.num_caps, kernel_size=5, padding=1, bias=False)
        self.bn_m = nn.BatchNorm2d(self.num_caps)

        self.conv_pose = nn.Conv2d(in_channels, self.num_caps*16, kernel_size=5, padding=1, bias=False)
        self.bn_pose = nn.BatchNorm2d(self.num_caps*16)

        self.emrouting = EmRouting2d(self.num_caps, self.num_caps, 16, kernel_size=3, stride=2, padding=1)
        self.bn_caps = nn.BatchNorm2d(self.num_caps*planes)

        self.conv_rec = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)
        self.bn_rec = nn.BatchNorm2d(64)

        self.conv_out4 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)
        self.bn_out4 = nn.BatchNorm2d(64)

        self.conv_out3 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)
        self.bn_out3 = nn.BatchNorm2d(64)

        self.conv_out2 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)
        self.bn_out2 = nn.BatchNorm2d(64)

        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)

        self.conv_pred5 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.conv_pred4 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.conv_pred3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.conv_pred2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        self.initialize()

    def forward(self, x2, x3, x4, x5):
        # primary caps
        m, pose = self.conv_m(x5), self.conv_pose(x5)
        m, pose = torch.sigmoid(self.bn_m(m)), self.bn_pose(pose)

        # caps
        m, pose = self.emrouting(m, pose)
        pose = self.bn_caps(pose)

        # reconstruction
        pose = self.leakyrelu(self.bn_rec(self.conv_rec(pose)))

        out5 = pose
        out5_up = F.interpolate(out5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        out4 = torch.cat((out5_up, x4), 1)
        out4 = self.dropout(self.leakyrelu(self.bn_out4(self.conv_out4(out4))))
        out4_up = F.interpolate(out4, size=x3.size()[2:], mode='bilinear', align_corners=True)
        out3 = torch.cat((out4_up, x3), 1)
        out3 = self.dropout(self.leakyrelu(self.bn_out3(self.conv_out3(out3))))
        out3_up = F.interpolate(out3, size=x2.size()[2:], mode='bilinear', align_corners=True)
        out2 = torch.cat((out3_up, x2), 1)
        out2 = self.dropout(self.leakyrelu(self.bn_out2(self.conv_out2(out2))))

        out5 = self.conv_pred5(out5)
        out4 = self.conv_pred4(out4)
        out3 = self.conv_pred3(out3)
        out2 = self.conv_pred2(out2)

        return out5, out4, out3, out2

    def initialize(self):
        weight_init(self)
