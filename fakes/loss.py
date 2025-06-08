"""import torch
import torch.nn.functional as F


def losses(b, a, T, margin, λ=0.7, mask=None, stop_gradient=False, teacher_weights=None):
    """
    b: List of List of teacher features → len(b) = num_teachers
       b[0]: Normal teacher
       b[1]: Abnormal teacher (opsiyonel)
    teacher_weights: Örn: [0.8, 0.2] → normal teacher daha ağır basar
    """
    if teacher_weights is None:
        teacher_weights = [1.0 / len(b)] * len(b)

    loss = 0.0
    for t_index, teacher_feat in enumerate(zip(b,a)):
        this_loss = 0.0
        margin_loss_n = 0.0
        margin_loss_a = 0.0
        contra_loss = 0.0

        for i in range(len(a)):
            s_ = a[i]
            t_ = teacher_feat[i].detach() if stop_gradient else teacher_feat[i]

            n, c, h, w = s_.shape

            s = s_.view(n, c, -1).transpose(1, 2)
            t = t_.view(n, c, -1).transpose(1, 2)

            s_norm = F.normalize(s, p=2, dim=2)
            t_norm = F.normalize(t, p=2, dim=2)

            cos_loss = 1 - F.cosine_similarity(s_norm, t_norm, dim=2)
            cos_loss = cos_loss.mean()

            simi = torch.matmul(s_norm, t_norm.transpose(1, 2)) / T
            simi = torch.exp(simi)
            simi_sum = simi.sum(dim=2, keepdim=True)
            simi = simi / (simi_sum + 1e-8)
            diag_sim = torch.diagonal(simi, dim1=1, dim2=2)

            if mask is None:
                contra_loss = -torch.log(diag_sim + 1e-8).mean()
                margin_loss_n = F.relu(margin - diag_sim).mean()
            else:
                if len(mask.size()) < 3:
                    normal_mask = (mask == 0)
                    abnormal_mask = (mask == 1)
                else:
                    mask_ = F.interpolate(mask, size=(h, w), mode='nearest').squeeze(1)
                    mask_flat = mask.view(mask_.size(0), -1)

                    normal_mask = (mask_flat == 0)
                    abnormal_mask = (mask_flat == 1)

                if normal_mask.sum() > 0:
                    diag_sim_normal = diag_sim[normal_mask]
                    contra_loss = -torch.log(diag_sim_normal + 1e-8).mean()
                    margin_loss_n = F.relu(margin - diag_sim_normal).mean()
                if abnormal_mask.sum() > 0:
                    diag_sim_abnormal = diag_sim[abnormal_mask]
                    margin_loss_a = F.relu(diag_sim_abnormal - margin / 2).mean()

            margin_loss = margin_loss_n + margin_loss_a
            this_loss += cos_loss * λ + contra_loss * (1 - λ) + margin_loss

        # Apply teacher-specific weight
        loss += teacher_weights[t_index] * this_loss

    return loss


def structure_loss(preds, mask, teacher_weights=None):
    """
    preds: List of predictions from different teachers
    mask: Ground truth segmentation mask
    teacher_weights: Örn. [0.8, 0.2] gibi ağırlıklar (toplamı 1 olmak zorunda değil)
    """
    if not isinstance(preds, list):
        preds = [preds]
    
    num_teachers = len(preds)
    if teacher_weights is None:
        teacher_weights = [1.0 / num_teachers] * num_teachers

    total_loss = 0.0
    for pred, w in zip(preds, teacher_weights):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / (weit.sum(dim=(2, 3)) + 1e-8)

        pred_sigmoid = torch.sigmoid(pred)
        inter = ((pred_sigmoid * mask) * weit).sum(dim=(2, 3))
        union = ((pred_sigmoid + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1 + 1e-8)

        loss = (wbce + wiou).mean()
        total_loss += w * loss

    return total_loss


if __name__=="__main__":
    tar_rep=([torch.rand(1,256,56,56).to("cuda"),torch.rand(1,512,28,28).to("cuda"),torch.rand(1,1024,14,14).to("cuda")]),([torch.rand(1,256,56,56).to("cuda"),torch.rand(1,512,28,28).to("cuda"),torch.rand(1,1024,14,14).to("cuda")])
    stu_rep=([torch.rand(1,256,56,56).to("cuda"),torch.rand(1,512,28,28).to("cuda"),torch.rand(1,1024,14,14).to("cuda")]),([torch.rand(1,256,56,56).to("cuda"),torch.rand(1,512,28,28).to("cuda"),torch.rand(1,1024,14,14).to("cuda")])
    model_loss=losses(tar_rep,stu_rep,0.1,1)
    print(model_loss)"""