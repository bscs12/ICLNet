import os
import sys
sys.path.insert(0, './')
sys.dont_write_bytecode = True
import cv2
from tqdm import tqdm
import metrics as M
import argparse
from model import dataset as Dataset
from torch.utils.data import DataLoader
from model.get_model import get_model
import torch
import numpy as np


def test(Dataset, args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    _MODEL_ = args.model
    _DATASET_ = args.dataset
    _CKPT_EPOCH_ = args.ckpt_epoch

    testset_path = 'datasets/'+_DATASET_+'/Test'
    pred_savepath = 'outputs/'+_MODEL_+'/prediction'+str(_CKPT_EPOCH_)
    checkpoint = 'outputs/'+_MODEL_+'/'+_MODEL_+str(_CKPT_EPOCH_)

    # data
    cfg = Dataset.Config(datapath=testset_path, checkpoint=checkpoint, mode='test')
    data = Dataset.Data(cfg, _MODEL_)
    loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=8)

    # model
    model = get_model(cfg, _MODEL_)
    model.train(False)
    model.cuda()

    with torch.no_grad():
        for image, (H, W), name in loader:
            image, shape = image.cuda().float(), (H, W)
            out5, out4, out3, out2 = model(image, shape)
            pred = torch.sigmoid(out2[0, 0]).cpu().numpy() * 255
            if not os.path.exists(pred_savepath):
                os.makedirs(pred_savepath)
            # cv2.imwrite(pred_savepath+'/'+name[0]+'.png', np.round(pred))
            cv2.imwrite(pred_savepath + '/' + name[0] + '.png', pred)

    Acc = M.Acc()
    Se = M.Se()
    Dice = M.Dice()
    IoU = M.IoU()
    HD = M.HD()

    gt_root = 'datasets/'+_DATASET_+'/Test/GT'
    pred_root = 'outputs/'+_MODEL_+'/prediction'+str(_CKPT_EPOCH_)
    gt_name_list = sorted(os.listdir(pred_savepath))

    for gt_name in tqdm(gt_name_list, total=len(gt_name_list)):
        gt_path = os.path.join(gt_root, gt_name)
        pred_path = os.path.join(pred_root, gt_name)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        if gt.shape != pred.shape:
            cv2.imwrite(pred_path, cv2.resize(pred, gt.shape[::-1]))

        Acc.step(pred=pred, gt=gt)
        Se.step(pred=pred, gt=gt)
        Dice.step(pred=pred, gt=gt)
        IoU.step(pred=pred, gt=gt)
        HD.step(pred=pred, gt=gt)

    Acc = Acc.get_results()['Acc']
    Se = Se.get_results()['Se']
    Dice = Dice.get_results()['Dice']
    IoU = IoU.get_results()['IoU']
    HD = HD.get_results()['HD']

    test_record = str(
        'Model:'+_MODEL_+'  ' +
        'Dataset:'+_DATASET_+'  ' +
        'ckpt_epoch:'+('%03d' % _CKPT_EPOCH_)+'  ' +
        'Acc:'+('%0.4f' % Acc)+'  ' +
        'Se:'+('%0.4f' % Se)+'  ' +
        'Dice:'+('%0.4f' % Dice)+'  ' +
        'IoU:'+('%0.4f' % IoU)+'  ' +
        'HD:'+('%0.4f' % HD)
         )

    print(test_record)

    txt = 'outputs/'+_MODEL_+'/test_record.txt'
    f = open(txt, 'a')
    f.write(test_record)
    f.write("\n")
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ICLNet-S')
    parser.add_argument('--dataset', type=str, default='AMUBUS')
    parser.add_argument('--test_all', type=bool, default=False)
    parser.add_argument('--ckpt_epoch', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    if not args.test_all:
        test(Dataset, args)
    else:
        for i in range(1, 60):
            args.ckpt_epoch = i
            checkpoint = 'outputs/'+args.model+'/'+args.model+str(args.ckpt_epoch)
            if os.path.exists(checkpoint):
                test(Dataset, args)
            i += 1
