import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from GradientReverseLayer import ReverseLayerF
from model import Attention, Decoder, DomainClassifier, Encoder, MSE, timeStepsShuffle


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, output_dim=32):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class SubjectAwareSupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SubjectAwareSupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels, subject_ids):
        device = features.device
        labels = labels.view(-1)
        subject_ids = subject_ids.view(-1)
        batch_size = features.size(0)

        if batch_size <= 1:
            return features.new_zeros(())

        logits = torch.matmul(features, features.t()) / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        label_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1))
        cross_subject_mask = ~subject_ids.unsqueeze(0).eq(subject_ids.unsqueeze(1))
        positive_mask = label_mask & cross_subject_mask & ~self_mask

        valid_anchor_mask = positive_mask.any(dim=1)
        if not valid_anchor_mask.any():
            return features.new_zeros(())

        exp_logits = torch.exp(logits) * (~self_mask).float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        positive_count = positive_mask.sum(dim=1).clamp(min=1).float()
        mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / positive_count
        loss = -mean_log_prob_pos[valid_anchor_mask].mean()
        return loss


class DMMRPreTrainingModelSupCon(nn.Module):
    def __init__(
        self,
        cuda,
        number_of_source=14,
        number_of_category=3,
        batch_size=10,
        time_steps=15,
        proj_hidden_dim=64,
        proj_output_dim=32,
        temperature=0.07,
    ):
        super(DMMRPreTrainingModelSupCon, self).__init__()
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.number_of_source = number_of_source
        self.attentionLayer = Attention(cuda, input_dim=310)
        self.sharedEncoder = Encoder(input_dim=310, hid_dim=64, n_layers=1)
        self.mse = MSE()
        self.domainClassifier = DomainClassifier(input_dim=64, output_dim=number_of_source)
        self.projector = ProjectionHead(64, proj_hidden_dim, proj_output_dim)
        self.supcon = SubjectAwareSupConLoss(temperature=temperature)
        for i in range(number_of_source):
            exec(
                "self.decoder{}=Decoder(input_dim=310, hid_dim=64, n_layers=1, output_dim=310)".format(i)
            )

    def encode(self, x, apply_noise=True):
        if apply_noise:
            x = timeStepsShuffle(x)
        x = self.attentionLayer(x, x.shape[0], self.time_steps)
        shared_last_out, shared_hn, shared_cn = self.sharedEncoder(x)
        return shared_last_out, shared_hn, shared_cn

    def adversarial_loss(self, shared_last_out, subject_id, m):
        reverse_feature = ReverseLayerF.apply(shared_last_out, m)
        subject_predict = self.domainClassifier(reverse_feature)
        subject_predict = F.log_softmax(subject_predict, dim=1)
        return F.nll_loss(subject_predict, subject_id)

    def contrastive_loss(self, shared_last_out, labels, subject_ids):
        projected = self.projector(shared_last_out)
        return self.supcon(projected, labels, subject_ids)

    def reconstruction_loss_from_encoded(self, shared_last_out, shared_hn, shared_cn, corres):
        corres = self.attentionLayer(corres, corres.shape[0], self.time_steps)
        splitted_tensors = torch.chunk(corres, self.number_of_source, dim=0)
        rec_loss = shared_last_out.new_zeros(())
        mix_subject_feature = 0
        for i in range(self.number_of_source):
            x_out, *_ = eval("self.decoder{}".format(i))(shared_last_out, shared_hn, shared_cn, self.time_steps)
            mix_subject_feature = mix_subject_feature + x_out
        shared_last_out_2, shared_hn_2, shared_cn_2 = self.sharedEncoder(mix_subject_feature)
        for i in range(self.number_of_source):
            x_out, *_ = eval("self.decoder{}".format(i))(shared_last_out_2, shared_hn_2, shared_cn_2, self.time_steps)
            rec_loss = rec_loss + self.mse(x_out, splitted_tensors[i])
        return rec_loss
