import os
import sys
sys.path.insert(0, './')
sys.dont_write_bytecode = True
import datetime
from model import dataset as Dataset
import argparse
import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from model.get_model import get_model
from utils.test import test


def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()


def train(Dataset, args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    _MODEL_ = args.model
    _DATASET_ = args.dataset
    _DARAPATH_ = 'datasets/'+args.dataset+'/Train'
    _LR_ = args.lr
    _DECAY_ = args.decay
    _MOMEN_ = args.momen
    _BATCH_SIZE_ = args.batch_size
    _NUM_EPOCHS_ = args.num_epochs
    _NUM_WORKERS_ = args.num_workers
    _SAVEPATH_ = 'outputs/'+args.model
    _TEST_WHILE_TRAINING_ = args.test_while_training

    print(args)

    # data
    cfg = Dataset.Config(datapath=_DARAPATH_, savepath=_SAVEPATH_, mode='train', batch_size=_BATCH_SIZE_, lr=_LR_, momen=_MOMEN_, decay=_DECAY_, num_epochs=_NUM_EPOCHS_)
    data = Dataset.Data(cfg, _MODEL_)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, num_workers=_NUM_WORKERS_, drop_last=True)

    # model
    model = get_model(cfg, _MODEL_)

    # parameter
    base, head = [], []
    for name, param in model.named_parameters():
        if 'encoder.conv1' in name or 'encoder.bn1' in name:
            pass
        elif 'encoder' in name:
            base.append(param)
        elif 'network' in name:
            base.append(param)
        else:
            head.append(param)

    # optimizer
    optimizer = torch.optim.SGD([{'params': base}, {'params': head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)

    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    sw = SummaryWriter(cfg.savepath)
    global_step = 0

    for epoch in range(cfg.num_epochs):
        model.train(True)
        model.cuda()
        optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.num_epochs+1)*2-1))*cfg.lr*0.1
        optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.num_epochs+1)*2-1))*cfg.lr

        for step, (image, mask) in enumerate(loader):
            image, mask = image.cuda(), mask.cuda()
            out5, out4, out3, out2 = model(image)

            loss1 = F.binary_cross_entropy_with_logits(out5, mask) + iou_loss(out5, mask)
            loss2 = F.binary_cross_entropy_with_logits(out4, mask) + iou_loss(out4, mask)
            loss3 = F.binary_cross_entropy_with_logits(out3, mask) + iou_loss(out3, mask)
            loss4 = F.binary_cross_entropy_with_logits(out2, mask) + iou_loss(out2, mask)
            loss = loss1 + loss2 + loss3 + loss4

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
            optimizer.step()

            # log
            global_step += 1
            sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'loss1': loss1.item(), 'loss2': loss2.item(), 'loss3': loss3.item(), 'loss4': loss4.item()}, global_step=global_step)
            if step % 10 == 0:
                print('%s  step:%06d/%03d/%03d  lr=%.6f  loss1=%.6f  loss2=%.6f  loss3=%.6f  loss4=%.6f' % (datetime.datetime.now(), global_step, epoch + 1, cfg.num_epochs, optimizer.param_groups[0]['lr'], loss1.item(), loss2.item(), loss3.item(), loss4.item()))

        # if (epoch + 1) > 30 and (epoch + 1) % 2 == 0:
        if (epoch + 1) % 2 == 0:
            # save model
            torch.save(model.state_dict(), _SAVEPATH_+'/'+_MODEL_+str(epoch+1))
            # test while training
            if _TEST_WHILE_TRAINING_:
                args.ckpt_epoch = epoch + 1
                test(Dataset, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ICLNet-R:ResNet50 | ICLNet-S:Swin | ICLNet-P:PVTv2 | ICLNet-V:VGG16 | ICLNet-M:CycleMLP
    parser.add_argument('--model', type=str, default='ICLNet-S')
    parser.add_argument('--dataset', type=str, default='AMUBUS')
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--momen', type=float, default=0.9)
    parser.add_argument('--decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--test_while_training', type=bool, default=True)
    parser.add_argument('--ckpt_epoch', type=int)
    args = parser.parse_args()
    train(Dataset, args)
