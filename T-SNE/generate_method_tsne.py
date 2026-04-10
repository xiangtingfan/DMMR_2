import argparse
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from preprocess import getDataLoaders
from model import DMMRPreTrainingModel, DMMRFineTuningModel, ModelReturnFeatures
from model_supcon import DMMRPreTrainingModelSupCon
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
    parser = argparse.ArgumentParser(description="Generate T-SNE plots for baseline or SupCon models")
    parser.add_argument("--dataset_name", type=str, default="seed4", choices=["seed3", "seed4"])
    parser.add_argument("--session", type=str, default="1")
    parser.add_argument("--subjects", type=int, default=15, choices=[15])
    parser.add_argument("--seed3_path", type=str, default="H:/SEED/feature_for_net_session{session}_LDS_de")
    parser.add_argument("--seed4_path", type=str, default="H:/SEED-IV/feature_for_net_session{session}_LDS_de")
    parser.add_argument("--num_workers_train", type=int, default=0)
    parser.add_argument("--num_workers_test", type=int, default=0)
    parser.add_argument("--one_subject", type=int, required=True, help="leave-one-subject-out target index, 0-based")
    parser.add_argument("--method", type=str, required=True, choices=["baseline", "supcon"])
    parser.add_argument("--model_way", type=str, required=True, help="model subdir under model/")
    parser.add_argument("--model_index", type=str, required=True, help="experiment index used when saving models")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--samples_per_subject", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument(
        "--views",
        type=str,
        default="tune_subject,tune_label",
        help="comma-separated subset of origin_subject,origin_label,pretrain_subject,pretrain_label,tune_subject,tune_label",
    )
    parser.add_argument("--proj_hidden_dim", type=int, default=64)
    parser.add_argument("--proj_output_dim", type=int, default=32)
    parser.add_argument("--supcon_temperature", type=float, default=0.07)
    args = parser.parse_args()

    args.source_subjects = args.subjects - 1
    if args.dataset_name == "seed3":
        args.path = args.seed3_path.format(session=args.session)
        args.cls_classes = 3
        args.time_steps = 30
        args.batch_size = 512
    else:
        args.path = args.seed4_path.format(session=args.session)
        args.cls_classes = 4
        args.time_steps = 10
        args.batch_size = 256
    return args


def _sample_rows(array_2d, sample_size):
    sample_size = min(sample_size, array_2d.shape[0])
    return np.random.choice(array_2d.shape[0], sample_size, replace=False)


def _build_models(args, source_loader):
    if args.method == "baseline":
        pretrain_model = DMMRPreTrainingModel(
            False,
            number_of_source=len(source_loader),
            number_of_category=args.cls_classes,
            batch_size=args.batch_size,
            time_steps=args.time_steps,
        )
    else:
        pretrain_model = DMMRPreTrainingModelSupCon(
            False,
            number_of_source=len(source_loader),
            number_of_category=args.cls_classes,
            batch_size=args.batch_size,
            time_steps=args.time_steps,
            proj_hidden_dim=args.proj_hidden_dim,
            proj_output_dim=args.proj_output_dim,
            temperature=args.supcon_temperature,
        )
    fine_tune_model = DMMRFineTuningModel(
        False,
        pretrain_model,
        number_of_source=len(source_loader),
        number_of_category=args.cls_classes,
        batch_size=args.batch_size,
        time_steps=args.time_steps,
    )
    return pretrain_model, fine_tune_model


def _load_models(args, source_loader):
    pretrain_model, fine_tune_model = _build_models(args, source_loader)
    pretrain_path = os.path.join("model", args.model_way, args.model_index, "{}_pretrain_model.pth".format(args.one_subject))
    tune_path = os.path.join("model", args.model_way, args.model_index, "{}_tune_model.pth".format(args.one_subject))
    if not os.path.exists(pretrain_path):
        raise FileNotFoundError("Missing checkpoint: {}".format(pretrain_path))
    if not os.path.exists(tune_path):
        raise FileNotFoundError("Missing checkpoint: {}".format(tune_path))
    pretrain_model.load_state_dict(torch.load(pretrain_path, map_location="cpu"))
    fine_tune_model.load_state_dict(torch.load(tune_path, map_location="cpu"))
    pretrain_model.eval()
    fine_tune_model.eval()
    return pretrain_model, fine_tune_model


def generate_tsne(data_loader_dict, args):
    source_loader = data_loader_dict["source_loader"]
    target_loader = data_loader_dict["test_loader"]
    pretrain_model, fine_tune_model = _load_models(args, source_loader)

    pretrain_return_feature = ModelReturnFeatures(pretrain_model, time_steps=args.time_steps)
    fine_tune_return_feature = ModelReturnFeatures(fine_tune_model, time_steps=args.time_steps)
    fine_tune_return_feature.eval()

    source_iters = [iter(loader) for loader in source_loader]
    origin_features_list = []
    origin_subject_id_list = []
    label_list = []
    pretrain_shared_features_list = []
    shared_features_list = []

    for subject_idx in range(len(source_iters)):
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
    view_set = {view.strip() for view in args.views.split(",") if view.strip()}
    if "origin_subject" in view_set:
        FeatureVisualize(origin_stacked_feature, stacked_subject_id).plot_tsne(os.path.join(args.output_dir, "origin_subject.jpg"))
    if "origin_label" in view_set:
        FeatureVisualize(origin_stacked_feature, stacked_label).plot_tsne(os.path.join(args.output_dir, "origin_label.jpg"))
    if "pretrain_subject" in view_set:
        FeatureVisualize(pretrain_shared_stacked_feature, stacked_subject_id).plot_tsne(os.path.join(args.output_dir, "pretrain_subject.jpg"))
    if "pretrain_label" in view_set:
        FeatureVisualize(pretrain_shared_stacked_feature, stacked_label).plot_tsne(os.path.join(args.output_dir, "pretrain_label.jpg"))
    if "tune_subject" in view_set:
        FeatureVisualize(shared_stacked_feature, stacked_subject_id).plot_tsne(os.path.join(args.output_dir, "tune_subject.jpg"))
    if "tune_label" in view_set:
        FeatureVisualize(shared_stacked_feature, stacked_label).plot_tsne(os.path.join(args.output_dir, "tune_label.jpg"))


if __name__ == "__main__":
    args = build_args()
    set_seed(args.seed)
    source_loaders, test_loader = getDataLoaders(args.one_subject, args)
    generate_tsne({"source_loader": source_loaders, "test_loader": test_loader}, args)
