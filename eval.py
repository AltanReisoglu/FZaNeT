from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from modules.loss import structure_loss
def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(1, 2))
    total = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2. * intersection + smooth) / (total + smooth)
    return 1 - dice.mean()


def evaluation_batch(c, model, dataloader, device):
    model.train_or_eval(type='eval')

    res=[]
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=True)

    with torch.no_grad():
        for i, sample in enumerate(progress_bar):
            img = sample[0].to(device)
            gt_mask = sample[2].to(device).float()
            _, _, pred_mask = model(img,max=False)
            loss = (
                structure_loss(pred_mask[0][0], gt_mask) +
                structure_loss(pred_mask[0][-1], gt_mask)
            ).item()  # .item() ile float'a çevir

            res.append(loss)
            progress_bar.set_postfix({
                "loss": f"{np.mean(res):.4f}",

            })

    return {
        "loss": np.mean(res),

    }
