import argparse
import os
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from preprocess import getDataLoaders
from train import trainDMMR


def set_seed(seed=3):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def tb_tag(*parts):
    raw = "_".join(str(part).strip() for part in parts if str(part).strip())
    return raw.replace(":", "").replace("/", "_").replace(" ", "_")


def main(data_loader_dict, args, optim_config, cuda, writer, one_subject, seed=3):
    set_seed(seed)
    if args.dataset_name == "seed3":
        iteration = 7
    elif args.dataset_name == "seed4":
        iteration = 3
    else:
        raise ValueError("Unsupported dataset_name: {}".format(args.dataset_name))
    acc = trainDMMR(data_loader_dict, optim_config, cuda, args, iteration, writer, one_subject)
    return acc


if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    parser = argparse.ArgumentParser(description="DMMR baseline quick comparison")

    parser.add_argument("--way", type=str, default="DMMR_BASELINE/seed4", help="name of current way")
    parser.add_argument("--index", type=str, default="baseline_seed4_3subjects", help="tensorboard index")
    parser.add_argument("--cuda_visible_devices", type=str, default="0", help="CUDA_VISIBLE_DEVICES value")

    parser.add_argument("--dataset_name", type=str, nargs="?", default="seed4", help="the dataset name, supporting seed3 and seed4")
    parser.add_argument("--session", type=str, nargs="?", default="1", help="selected session")
    parser.add_argument("--subjects", type=int, choices=[15], default=15, help="the number of all subject")
    parser.add_argument("--max_subjects", type=int, default=3, help="number of held-out subjects to run for quick comparison")
    parser.add_argument("--dim", type=int, default=310, help="dim of input")
    parser.add_argument("--seed3_path", type=str, default="H:/SEED/feature_for_net_session{session}_LDS_de", help="path to local SEED .mat files")
    parser.add_argument("--seed4_path", type=str, default="H:/SEED-IV/feature_for_net_session{session}_LDS_de", help="path to local SEED-IV .mat files")
    parser.add_argument("--num_workers_train", type=int, default=None, help="number of dataloader workers for training")
    parser.add_argument("--num_workers_test", type=int, default=None, help="number of dataloader workers for testing")

    parser.add_argument("--input_dim", type=int, default=310, help="input dim is the same with sample's last dim")
    parser.add_argument("--hid_dim", type=int, default=64, help="hid dim is for hidden layer of lstm")
    parser.add_argument("--n_layers", type=int, default=1, help="num of layers of lstm")
    parser.add_argument("--epoch_fineTuning", type=int, default=500, help="epoch of the fine-tuning phase")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="weight decay")
    parser.add_argument("--beta", type=float, default=0.05, help="balancing hyperparameter in the loss of pretraining phase")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    args.source_subjects = args.subjects - 1

    if args.num_workers_train is None or args.num_workers_test is None:
        if os.name == "nt":
            default_num_workers_train = 0
            default_num_workers_test = 0
        elif cuda:
            default_num_workers_train = 4
            default_num_workers_test = 2
        else:
            default_num_workers_train = 0
            default_num_workers_test = 0
        if args.num_workers_train is None:
            args.num_workers_train = default_num_workers_train
        if args.num_workers_test is None:
            args.num_workers_test = default_num_workers_test

    if args.dataset_name == "seed3":
        args.path = args.seed3_path.format(session=args.session)
        args.cls_classes = 3
        args.time_steps = 30
        args.batch_size = 512
        args.epoch_preTraining = 300
    elif args.dataset_name == "seed4":
        args.path = args.seed4_path.format(session=args.session)
        args.cls_classes = 4
        args.time_steps = 10
        args.batch_size = 256
        args.epoch_preTraining = 400
    else:
        raise ValueError("need to define the input dataset")

    optim_config = {"lr": args.lr, "weight_decay": args.weight_decay}
    acc_list = []
    writer = SummaryWriter("data/session" + args.session + "/" + args.way + "/" + args.index)
    subjects_to_run = min(args.max_subjects, args.subjects)

    for one_subject in range(subjects_to_run):
        source_loaders, test_loader = getDataLoaders(one_subject, args)
        data_loader_dict = {"source_loader": source_loaders, "test_loader": test_loader}
        acc = main(data_loader_dict, args, optim_config, cuda, writer, one_subject)
        writer.add_scalars(tb_tag("single experiment acc"), {"test acc": acc}, one_subject + 1)
        writer.flush()
        acc_list.append(acc)

    writer.add_text("final acc avg", str(np.average(acc_list)))
    writer.add_text("final acc std", str(np.std(acc_list)))
    writer.add_text("final each acc", ",".join(str(x) for x in acc_list))
    writer.add_scalars(tb_tag("final experiment acc scala", "avg"), {"test acc": np.average(acc_list)})
    writer.add_scalars(tb_tag("final experiment acc scala", "std"), {"test acc": np.std(acc_list)})
    writer.close()
