import cv2
import numpy as np
import torch

_EPS = 1e-16
_TYPE = np.float64


def _prepare_data(pred: np.ndarray, gt: np.ndarray) -> tuple:
    gt = gt > 128
    pred = pred / 255
    if pred.max() != pred.min():
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    return pred, gt


class Acc(object):
    def __init__(self, threshold=0.5):
        self.Acc = []
        self.threshold = threshold

    def step(self, pred: np.ndarray, gt: torch.tensor):
        pred, gt = _prepare_data(pred=pred, gt=gt)

        pred = torch.from_numpy(pred)
        gt = torch.from_numpy(gt)

        SR = pred.view(-1)
        GT = gt.view(-1)

        SR = SR.data.cpu().numpy()
        GT = GT.data.cpu().numpy()

        SR = SR > self.threshold
        GT = GT == np.max(GT)
        corr = np.sum(SR == GT)

        Acc = float(corr) / float(SR.shape[0])

        self.Acc.append(Acc)

    def get_results(self) -> dict:
        Acc = np.mean(np.array(self.Acc, dtype=_TYPE))
        return dict(Acc=Acc)


class Se(object):
    def __init__(self, threshold=0.5):
        self.Se = []
        self.threshold = threshold

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)

        SR = (pred > self.threshold).astype(np.float32)
        GT = (gt == np.max(gt)).astype(np.float32)

        # TP: True Positive
        # FN: False Negative
        TP = (((SR == 1.).astype(np.float32) + (GT == 1.).astype(np.float32)) == 2.).astype(np.float32)
        FN = (((SR == 0.).astype(np.float32) + (GT == 1.).astype(np.float32)) == 2.).astype(np.float32)

        Se = float(np.sum(TP)) / (float(np.sum(TP + FN)) + 1e-6)

        self.Se.append(Se)

    def get_results(self) -> dict:
        Se = np.mean(np.array(self.Se, dtype=_TYPE))
        return dict(Se=Se)


class Dice(object):
    def __init__(self, threshold=0.5):
        self.Dice = []
        self.threshold = threshold

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)

        SR = (pred > self.threshold).astype(np.float32)
        GT = (gt == np.max(gt)).astype(np.float32)

        Inter = np.sum(((SR + GT) == 2).astype(np.float32))
        Dice = float(2 * Inter) / (float(np.sum(SR) + np.sum(GT)) + 1e-6)

        self.Dice.append(Dice)

    def get_results(self) -> dict:
        Dice = np.mean(np.array(self.Dice, dtype=_TYPE))
        return dict(Dice=Dice)


class IoU(object):
    def __init__(self, threshold=0.5):
        self.IoU = []
        self.threshold = threshold

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)

        SR = (pred > self.threshold).astype(np.float32)
        GT = (gt == np.max(gt)).astype(np.float32)

        TP = (((SR == 1.).astype(np.float32) + (GT == 1.).astype(np.float32)) == 2.).astype(np.float32)
        FP = (((SR == 1.).astype(np.float32) + (GT == 0.).astype(np.float32)) == 2.).astype(np.float32)
        FN = (((SR == 0.).astype(np.float32) + (GT == 1.).astype(np.float32)) == 2.).astype(np.float32)

        IoU = float(np.sum(TP)) / (float(np.sum(TP + FP + FN)) + 1e-6)

        self.IoU.append(IoU)

    def get_results(self) -> dict:
            IoU = np.mean(np.array(self.IoU, dtype=_TYPE))
            return dict(IoU=IoU)


class HD(object):
    def __init__(self):
        self.HD = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred = pred / 255
        gt = gt / 255
        pred = cv2.resize(pred, (256, 256))
        gt = cv2.resize(gt, (256, 256))
        # pred, gt = _prepare_data(pred=pred, gt=gt)
        pred = pred > 0.65
        gt = gt > 0.65
        x = torch.from_numpy(pred).float()
        y = torch.from_numpy(gt).float()
        x = x.unsqueeze(0).unsqueeze(0)
        y = y.unsqueeze(0).unsqueeze(0)
        distance_matrix = torch.cdist(x, y, p=2)  # p=2 means Euclidean Distance
        value1 = distance_matrix.min(2)[0].max(1, keepdim=True)[0]
        value2 = distance_matrix.min(1)[0].max(1, keepdim=True)[0]
        value = torch.cat((value1, value2), dim=1)
        HD = value.max(1)[0].mean().item()
        self.HD.append(HD)

    def get_results(self) -> dict:
        HD = np.mean(np.array(self.HD, dtype=_TYPE))
        return dict(HD=HD)
