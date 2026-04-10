import argparse
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.manifold import TSNE

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from preprocess import getDataLoaders
from model import DMMRPreTrainingModel, DMMRFineTuningModel, ModelReturnFeatures
from model_supcon import DMMRPreTrainingModelSupCon


def set_seed(seed=3):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def build_args():
    parser = argparse.ArgumentParser(description="Generate clean T-SNE comparison panels")
    parser.add_argument("--dataset_name", type=str, default="seed4", choices=["seed3", "seed4"])
    parser.add_argument("--session", type=str, default="1")
    parser.add_argument("--subjects", type=int, default=15, choices=[15])
    parser.add_argument("--seed3_path", type=str, default="H:/SEED/feature_for_net_session{session}_LDS_de")
    parser.add_argument("--seed4_path", type=str, default="H:/SEED-IV/feature_for_net_session{session}_LDS_de")
    parser.add_argument("--num_workers_train", type=int, default=0)
    parser.add_argument("--num_workers_test", type=int, default=0)
    parser.add_argument("--samples_per_subject", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3)
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


def _build_models(method, args, source_loader):
    if method == "baseline":
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


def _load_models(method, model_way, model_index, one_subject, args, source_loader):
    pretrain_model, fine_tune_model = _build_models(method, args, source_loader)
    pretrain_path = os.path.join(ROOT_DIR, "model", model_way, model_index, f"{one_subject}_pretrain_model.pth")
    tune_path = os.path.join(ROOT_DIR, "model", model_way, model_index, f"{one_subject}_tune_model.pth")
    pretrain_model.load_state_dict(torch.load(pretrain_path, map_location="cpu"))
    fine_tune_model.load_state_dict(torch.load(tune_path, map_location="cpu"))
    pretrain_model.eval()
    fine_tune_model.eval()
    return pretrain_model, fine_tune_model


def extract_feature_sets(method, model_way, model_index, one_subject, args):
    source_loaders, test_loader = getDataLoaders(one_subject, args)
    pretrain_model, fine_tune_model = _load_models(method, model_way, model_index, one_subject, args, source_loaders)
    pretrain_return_feature = ModelReturnFeatures(pretrain_model, time_steps=args.time_steps)
    fine_tune_return_feature = ModelReturnFeatures(fine_tune_model, time_steps=args.time_steps)
    source_iters = [iter(loader) for loader in source_loaders]

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

    target_data, target_label = next(iter(test_loader))
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

    return {
        "origin": (
            np.concatenate(origin_features_list, axis=0),
            np.concatenate(origin_subject_id_list, axis=0),
            np.concatenate(label_list, axis=0),
        ),
        "pretrain": (
            np.concatenate(pretrain_shared_features_list, axis=0),
            np.concatenate(origin_subject_id_list, axis=0),
            np.concatenate(label_list, axis=0),
        ),
        "tune": (
            np.concatenate(shared_features_list, axis=0),
            np.concatenate(origin_subject_id_list, axis=0),
            np.concatenate(label_list, axis=0),
        ),
    }


def save_tsne_panel(features, labels, out_path, title):
    tsne = TSNE(n_components=2, init="pca", random_state=0)
    transformed = tsne.fit_transform(features)
    x_min, x_max = np.min(transformed, axis=0), np.max(transformed, axis=0)
    normalized = (transformed - x_min) / (x_max - x_min + 1e-12)
    colors = plt.cm.tab20.colors

    plt.figure(figsize=(5.2, 4.6))
    for idx in range(normalized.shape[0]):
        plt.scatter(
            normalized[idx, 0],
            normalized[idx, 1],
            color=colors[int(labels[idx]) % len(colors)],
            s=10,
            alpha=0.85,
        )
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def compose_grid(entries, out_path, footer_text):
    imgs = [(title, Image.open(path).convert("RGB")) for title, path in entries]
    cell_w = max(img.size[0] for _, img in imgs)
    cell_h = max(img.size[1] for _, img in imgs)
    header_h = 34
    footer_h = 42
    pad = 16
    cols = 4
    rows = 2
    canvas_w = cols * cell_w + (cols + 1) * pad
    canvas_h = rows * (cell_h + header_h) + (rows + 1) * pad + footer_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for idx, (title, img) in enumerate(imgs):
        r = idx // cols
        c = idx % cols
        x = pad + c * (cell_w + pad)
        y = pad + r * (cell_h + header_h + pad)
        draw.text((x, y), title, fill="black", font=font)
        ix = x + (cell_w - img.size[0]) // 2
        iy = y + header_h
        canvas.paste(img, (ix, iy))

    draw.text((pad, canvas_h - footer_h + 10), footer_text, fill="black", font=font)
    canvas.save(out_path, quality=95)


def main():
    args = build_args()
    set_seed(args.seed)

    comparisons = {
        1: {
            "baseline_acc": 0.8046875,
            "supcon_acc": 0.669921875,
            "label": "subject2",
        },
        8: {
            "baseline_acc": 0.611328125,
            "supcon_acc": 0.740234375,
            "label": "subject9",
        },
    }
    column_titles = [
        "Baseline-Subject",
        "Baseline-Label",
        "SupCon-Subject",
        "SupCon-Label",
    ]
    method_cfg = {
        "baseline": ("DMMR/seed4", "1"),
        "supcon": ("DMMR_SUPCON/seed4", "supcon_seed4_3subjects"),
    }
    output_root = os.path.join(ROOT_DIR, "T-SNE", "plot", "seed4", "session1", "comparison_clean")
    os.makedirs(output_root, exist_ok=True)

    extracted = {}
    for subject_idx in comparisons:
        for method, (model_way, model_index) in method_cfg.items():
            extracted[(subject_idx, method)] = extract_feature_sets(method, model_way, model_index, subject_idx, args)

    for stage in ["origin", "pretrain", "tune"]:
        entries = []
        for subject_idx in [1, 8]:
            info = comparisons[subject_idx]
            for method in ["baseline", "supcon"]:
                stage_features, stage_subject_ids, stage_labels = extracted[(subject_idx, method)][stage]
                for title_suffix, labels in [("Subject", stage_subject_ids), ("Label", stage_labels)]:
                    file_name = f"{info['label']}_{method}_{stage}_{title_suffix.lower()}.jpg"
                    out_path = os.path.join(output_root, file_name)
                    panel_title = f"{info['label']} | {method} | {stage} | {title_suffix}"
                    save_tsne_panel(stage_features, labels, out_path, panel_title)
                    col_title = column_titles[(0 if method == "baseline" else 2) + (0 if title_suffix == "Subject" else 1)]
                    row_title = "Subject 2" if subject_idx == 1 else "Subject 9"
                    entries.append((f"{row_title} | {col_title}", out_path))

        footer_text = (
            "Accuracy change: Subject 2 baseline 0.8047 -> supcon 0.6699 (-0.1348); "
            "Subject 9 baseline 0.6113 -> supcon 0.7402 (+0.1289)"
        )
        compose_grid(
            entries,
            os.path.join(output_root, f"{stage}_comparison_grid.jpg"),
            footer_text,
        )


if __name__ == "__main__":
    main()
