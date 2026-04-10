from preprocess import getDataLoaders
import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

from model import DMMRPreTrainingModel, DMMRFineTuningModel, ModelReturnFeatures
from train import FeatureVisualize


def set_seed(seed=3):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def build_args():
    parser = argparse.ArgumentParser(description="Generate T-SNE plots for DMMR")
    parser.add_argument("--dataset_name", type=str, default="seed3", choices=["seed3", "seed4"])
    parser.add_argument("--session", type=str, default="1")
    parser.add_argument("--subjects", type=int, default=15, choices=[15])
    parser.add_argument("--seed3_path", type=str, default="H:/SEED/feature_for_net_session{session}_LDS_de")
    parser.add_argument("--seed4_path", type=str, default="H:/SEED-IV/feature_for_net_session{session}_LDS_de")
    parser.add_argument("--cuda_visible_devices", type=str, default="0")
    parser.add_argument("--num_workers_train", type=int, default=0)
    parser.add_argument("--num_workers_test", type=int, default=0)
    parser.add_argument("--one_subject", type=int, default=0, help="leave-one-subject-out target index, 0-based")
    parser.add_argument("--model_way", type=str, default=None, help="model subdir under model/, e.g. DMMR/seed3")
    parser.add_argument("--model_index", type=str, default="0", help="experiment index used when saving models")
    parser.add_argument("--output_dir", type=str, default=None, help="directory for generated plots")
    parser.add_argument("--samples_per_subject", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    args.source_subjects = args.subjects - 1
    if args.dataset_name == "seed3":
        args.path = args.seed3_path.format(session=args.session)
        args.cls_classes = 3
        args.time_steps = 30
        args.batch_size = 512
        default_way = "DMMR/seed3"
    else:
        args.path = args.seed4_path.format(session=args.session)
        args.cls_classes = 4
        args.time_steps = 10
        args.batch_size = 256
        default_way = "DMMR/seed4"

    if args.model_way is None:
        args.model_way = default_way
    if args.output_dir is None:
        args.output_dir = os.path.join("T-SNE", "plot", args.dataset_name, "session{}".format(args.session), args.model_index)
    return args


def _sample_rows(array_2d, sample_size):
    sample_size = min(sample_size, array_2d.shape[0])
    return np.random.choice(array_2d.shape[0], sample_size, replace=False)


def generate_tsne(data_loader_dict, args):
    source_loader = data_loader_dict["source_loader"]
    target_loader = data_loader_dict["test_loader"]

    pretrain_model = DMMRPreTrainingModel(
        False,
        number_of_source=len(source_loader),
        number_of_category=args.cls_classes,
        batch_size=args.batch_size,
        time_steps=args.time_steps,
    )
    pretrain_path = os.path.join("model", args.model_way, args.model_index, "{}_pretrain_model.pth".format(args.one_subject))
    tune_path = os.path.join("model", args.model_way, args.model_index, "{}_tune_model.pth".format(args.one_subject))
    pretrain_model.load_state_dict(torch.load(pretrain_path, map_location="cpu"))
    pretrain_model.eval()

    pretrain_return_feature = ModelReturnFeatures(pretrain_model, time_steps=args.time_steps)

    fine_tune_model = DMMRFineTuningModel(
        False,
        pretrain_model,
        number_of_source=len(source_loader),
        number_of_category=args.cls_classes,
        batch_size=args.batch_size,
        time_steps=args.time_steps,
    )
    fine_tune_model.load_state_dict(torch.load(tune_path, map_location="cpu"))
    fine_tune_model.eval()
    fine_tune_return_feature = ModelReturnFeatures(fine_tune_model, time_steps=args.time_steps)
    fine_tune_return_feature.eval()

    source_iters = [iter(loader) for loader in source_loader]
    origin_features_list = []
    origin_subject_id_list = []
    label_list = []
    pretrain_shared_features_list = []
    shared_features_list = []

    for subject_idx in range(len(source_iters)):
        try:
            source_data, source_label = next(source_iters[subject_idx])
        except StopIteration:
            source_iters[subject_idx] = iter(source_loader[subject_idx])
            source_data, source_label = next(source_iters[subject_idx])

        _, pretrain_shared_feature = pretrain_return_feature(source_data)
        _, shared_feature = fine_tune_return_feature(source_data)

        source_data_narray = source_data.numpy().reshape(source_data.shape[0], -1)
        source_label_narray = source_label.squeeze().numpy()
        random_indices = _sample_rows(source_data_narray, args.samples_per_subject)

        origin_features_list.append(source_data_narray[random_indices])
        origin_subject_id_list.append(np.full((len(random_indices),), subject_idx))
        label_list.append(source_label_narray[random_indices])
        pretrain_shared_features_list.append(pretrain_shared_feature.detach().numpy()[random_indices])
        shared_features_list.append(shared_feature.detach().numpy()[random_indices])

    target_data, target_label = next(iter(target_loader))
    _, target_pretrain_shared_feature = pretrain_return_feature(target_data)
    _, target_shared_feature = fine_tune_return_feature(target_data)
    target_data_narray = target_data.numpy().reshape(target_data.shape[0], -1)
    target_label_narray = target_label.squeeze().numpy()
    target_indices = _sample_rows(target_data_narray, args.samples_per_subject)

    origin_features_list.append(target_data_narray[target_indices])
    origin_subject_id_list.append(np.full((len(target_indices),), len(source_iters)))
    label_list.append(target_label_narray[target_indices])
    pretrain_shared_features_list.append(target_pretrain_shared_feature.detach().numpy()[target_indices])
    shared_features_list.append(target_shared_feature.detach().numpy()[target_indices])

    origin_stacked_feature = np.concatenate(origin_features_list, axis=0)
    stacked_subject_id = np.concatenate(origin_subject_id_list, axis=0)
    stacked_label = np.concatenate(label_list, axis=0)
    pretrain_shared_stacked_feature = np.concatenate(pretrain_shared_features_list, axis=0)
    shared_stacked_feature = np.concatenate(shared_features_list, axis=0)

    os.makedirs(args.output_dir, exist_ok=True)
    FeatureVisualize(origin_stacked_feature, stacked_subject_id).plot_tsne(os.path.join(args.output_dir, "origin_subject.jpg"))
    FeatureVisualize(origin_stacked_feature, stacked_label).plot_tsne(os.path.join(args.output_dir, "origin_label.jpg"))
    FeatureVisualize(pretrain_shared_stacked_feature, stacked_subject_id).plot_tsne(os.path.join(args.output_dir, "pretrain_subject.jpg"))
    FeatureVisualize(pretrain_shared_stacked_feature, stacked_label).plot_tsne(os.path.join(args.output_dir, "pretrain_label.jpg"))
    FeatureVisualize(shared_stacked_feature, stacked_subject_id).plot_tsne(os.path.join(args.output_dir, "tune_subject.jpg"))
    FeatureVisualize(shared_stacked_feature, stacked_label).plot_tsne(os.path.join(args.output_dir, "tune_label.jpg"))


if __name__ == "__main__":
    args = build_args()
    set_seed(args.seed)
    source_loaders, test_loader = getDataLoaders(args.one_subject, args)
    generate_tsne({"source_loader": source_loaders, "test_loader": test_loader}, args)
