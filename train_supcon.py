import copy
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch

from model import DMMRFineTuningModel, DMMRTestModel
from model_supcon import DMMRPreTrainingModelSupCon
from test import testDMMR


def tb_tag(*parts):
    raw = "_".join(str(part).strip() for part in parts if str(part).strip())
    return raw.replace(":", "").replace("/", "_").replace(" ", "_")


def _fetch_source_batches(source_iters, source_loader):
    batch_dict = defaultdict(list)
    data_dict = defaultdict(list)
    label_dict = defaultdict(list)

    for subject_idx in range(len(source_iters)):
        try:
            batch_dict[subject_idx] = next(source_iters[subject_idx])
        except StopIteration:
            source_iters[subject_idx] = iter(source_loader[subject_idx])
            batch_dict[subject_idx] = next(source_iters[subject_idx])

        source_data, source_label = batch_dict[subject_idx]
        for index, label_tensor in enumerate(source_label):
            cur_label = label_tensor[0].item()
            data_dict[subject_idx].append(source_data[index])
            label_dict[subject_idx].append(cur_label)
    return batch_dict, data_dict, label_dict


def _build_correspondence_batch(source_label, data_dict, label_dict):
    label_pools = _build_label_pools(data_dict, label_dict)
    return _build_correspondence_batch_from_pools(source_label, label_pools)


def _build_label_pools(data_dict, label_dict):
    label_pools = []
    for subject_idx in range(len(data_dict)):
        current_pool = defaultdict(list)
        cur_data_list = data_dict[subject_idx]
        cur_label_list = label_dict[subject_idx]
        for sample, sample_label in zip(cur_data_list, cur_label_list):
            current_pool[sample_label].append(sample)
        label_pools.append(current_pool)
    return label_pools


def _build_correspondence_batch_from_pools(source_label, label_pools):
    corres_batch_data = []
    for label_tensor in source_label:
        label_cur = label_tensor[0].item()
        for subject_pool in label_pools:
            if not subject_pool[label_cur]:
                raise RuntimeError("No same-label sample found while building corres_batch_data.")
            corres_batch_data.append(random.choice(subject_pool[label_cur]))
    return torch.stack(corres_batch_data)


def _stack_joint_batch(batch_dict):
    all_source_data = []
    all_source_label = []
    all_subject_id = []

    for subject_idx, (source_data, source_label) in batch_dict.items():
        all_source_data.append(source_data)
        all_source_label.append(source_label.squeeze(1))
        all_subject_id.append(
            torch.full((source_data.size(0),), subject_idx, dtype=torch.long)
        )

    x_all = torch.cat(all_source_data, dim=0)
    y_all = torch.cat(all_source_label, dim=0)
    d_all = torch.cat(all_subject_id, dim=0)
    return x_all, y_all, d_all


def trainDMMR_supcon(data_loader_dict, optimizer_config, cuda, args, iteration, writer, one_subject):
    source_loader = data_loader_dict["source_loader"]
    preTrainModel = DMMRPreTrainingModelSupCon(
        cuda,
        number_of_source=len(source_loader),
        number_of_category=args.cls_classes,
        batch_size=args.batch_size,
        time_steps=args.time_steps,
        proj_hidden_dim=args.proj_hidden_dim,
        proj_output_dim=args.proj_output_dim,
        temperature=args.supcon_temperature,
    )
    if cuda:
        preTrainModel = preTrainModel.cuda()

    source_iters = [iter(loader) for loader in source_loader]
    optimizer_PreTraining = torch.optim.Adam(preTrainModel.parameters(), **optimizer_config)

    acc_final = 0
    for epoch in range(args.epoch_preTraining):
        print("epoch: " + str(epoch))
        start_time_pretrain = time.time()
        preTrainModel.train()
        data_set_all = 0

        for step in range(1, iteration + 1):
            p = float(step + epoch * iteration) / args.epoch_preTraining / iteration
            m = 2.0 / (1.0 + np.exp(-10 * p)) - 1
            batch_dict, data_dict, label_dict = _fetch_source_batches(source_iters, source_loader)
            label_pools = _build_label_pools(data_dict, label_dict)

            optimizer_PreTraining.zero_grad()

            rec_loss_total = next(preTrainModel.parameters()).new_zeros(())
            shared_features_all = []
            labels_all = []
            subject_ids_all = []
            for subject_idx in range(len(source_iters)):
                source_data, source_label = batch_dict[subject_idx]
                corres_batch_data = _build_correspondence_batch_from_pools(source_label, label_pools)
                subject_ids = torch.full((source_data.size(0),), subject_idx, dtype=torch.long)
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                    corres_batch_data = corres_batch_data.cuda()
                    subject_ids = subject_ids.cuda()
                shared_src, shared_hn, shared_cn = preTrainModel.encode(source_data, apply_noise=True)
                shared_features_all.append(shared_src)
                labels_all.append(source_label.squeeze(1))
                subject_ids_all.append(subject_ids)
                rec_loss_total = rec_loss_total + preTrainModel.reconstruction_loss_from_encoded(
                    shared_src, shared_hn, shared_cn, corres_batch_data
                )
                data_set_all += len(source_label)

            shared_all = torch.cat(shared_features_all, dim=0)
            y_all = torch.cat(labels_all, dim=0)
            d_all = torch.cat(subject_ids_all, dim=0)
            adv_loss = preTrainModel.adversarial_loss(shared_all, d_all, m)
            con_loss = preTrainModel.contrastive_loss(shared_all, y_all, d_all)

            loss_pretrain = rec_loss_total + args.beta * adv_loss + args.lambda_con * con_loss
            loss_pretrain.backward()
            optimizer_PreTraining.step()

        print("data set amount: " + str(data_set_all))
        writer.add_scalars(
            tb_tag("subject", one_subject + 1, "train DMMR_SUPCON/loss"),
            {
                "loss_pretrain": loss_pretrain.data,
                "rec_loss": rec_loss_total.data,
                "adv_loss": adv_loss.data,
                "con_loss": con_loss.data,
            },
            epoch + 1,
        )
        writer.add_scalar(
            tb_tag("subject", one_subject + 1, "train DMMR_SUPCON/pretrain total loss"),
            loss_pretrain.item(),
            epoch + 1,
        )
        writer.add_scalar(
            tb_tag("subject", one_subject + 1, "train DMMR_SUPCON/pretrain rec loss"),
            rec_loss_total.item(),
            epoch + 1,
        )
        writer.add_scalar(
            tb_tag("subject", one_subject + 1, "train DMMR_SUPCON/pretrain adv loss"),
            adv_loss.item(),
            epoch + 1,
        )
        writer.add_scalar(
            tb_tag("subject", one_subject + 1, "train DMMR_SUPCON/pretrain con loss"),
            con_loss.item(),
            epoch + 1,
        )
        end_time_pretrain = time.time()
        pretrain_epoch_time = end_time_pretrain - start_time_pretrain
        print("The time required for one pre-training epoch is:", pretrain_epoch_time, "second")
        print("rec_loss: " + str(rec_loss_total))
        print("adv_loss: " + str(adv_loss))
        print("con_loss: " + str(con_loss))

    source_iters2 = [iter(loader) for loader in source_loader]
    fineTuneModel = DMMRFineTuningModel(
        cuda,
        preTrainModel,
        number_of_source=len(source_loader),
        number_of_category=args.cls_classes,
        batch_size=args.batch_size,
        time_steps=args.time_steps,
    )
    optimizer_FineTuning = torch.optim.Adam(fineTuneModel.parameters(), **optimizer_config)
    if cuda:
        fineTuneModel = fineTuneModel.cuda()

    best_pretrain_model = copy.deepcopy(preTrainModel.state_dict())
    best_tune_model = copy.deepcopy(fineTuneModel.state_dict())
    best_test_model = copy.deepcopy(DMMRTestModel(fineTuneModel).state_dict())

    for epoch in range(args.epoch_fineTuning):
        print("epoch: " + str(epoch))
        start_time = time.time()
        fineTuneModel.train()
        count = 0
        data_set_all = 0

        for _ in range(1, iteration + 1):
            batch_dict, _, _ = _fetch_source_batches(source_iters2, source_loader)
            for subject_idx in range(len(source_iters2)):
                source_data, source_label = batch_dict[subject_idx]
                if cuda:
                    source_data = source_data.cuda()
                    source_label = source_label.cuda()
                data_set_all += len(source_label)
                optimizer_FineTuning.zero_grad()
                x_pred, x_logits, cls_loss = fineTuneModel(source_data, source_label)
                cls_loss.backward()
                optimizer_FineTuning.step()
                _, pred = torch.max(x_pred, dim=1)
                count += pred.eq(source_label.squeeze().data.view_as(pred)).sum()

        end_time = time.time()
        epoch_time = end_time - start_time
        print("The time required for one fine-tuning epoch is:", epoch_time, "second")
        print("data set amount: " + str(data_set_all))
        acc = float(count) / data_set_all
        writer.add_scalars(
            tb_tag("subject", one_subject + 1, "train DMMR_SUPCON/loss"),
            {"cls_loss": cls_loss.data},
            epoch + 1,
        )
        writer.add_scalar(
            tb_tag("subject", one_subject + 1, "train DMMR_SUPCON/fine-tune cls loss"),
            cls_loss.item(),
            epoch + 1,
        )
        writer.add_scalar(tb_tag("subject", one_subject + 1, "train DMMR_SUPCON/train accuracy"), acc, epoch + 1)
        print("acc: " + str(acc))

        testModel = DMMRTestModel(fineTuneModel)
        acc_DMMR = testDMMR(data_loader_dict["test_loader"], testModel, cuda, args.batch_size)
        print("acc_DMMR: " + str(acc_DMMR))
        writer.add_scalars(
            tb_tag("subject", one_subject + 1, "test DMMR_SUPCON/test acc"),
            {"test acc": acc_DMMR},
            epoch + 1,
        )
        if acc_DMMR > acc_final:
            acc_final = acc_DMMR
            best_pretrain_model = copy.deepcopy(preTrainModel.state_dict())
            best_tune_model = copy.deepcopy(fineTuneModel.state_dict())
            best_test_model = copy.deepcopy(testModel.state_dict())

    writer.add_text(
        tb_tag("subject", one_subject + 1, "summary"),
        "best_test_acc={:.6f}".format(acc_final),
    )

    modelDir = "model/" + args.way + "/" + args.index + "/"
    os.makedirs(modelDir, exist_ok=True)
    torch.save(best_pretrain_model, modelDir + str(one_subject) + "_pretrain_model.pth")
    torch.save(best_tune_model, modelDir + str(one_subject) + "_tune_model.pth")
    torch.save(best_test_model, modelDir + str(one_subject) + "_test_model.pth")
    return acc_final
