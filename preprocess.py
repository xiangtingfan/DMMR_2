import os

import numpy as np
import scipy.io as scio
import torch


SEED_VIDEO_TIME = [235, 233, 206, 238, 185, 195, 237, 216, 265, 237, 235, 233, 235, 238, 206]
SEEDIV_VIDEO_TIME = [42, 23, 49, 32, 22, 40, 38, 52, 36, 42, 12, 27, 54, 42, 64, 35, 17, 44, 35, 12, 28, 28, 43, 34]


def get_data_path(file_path):
    data_path = []
    for f in os.listdir(file_path):
        if f.startswith("."):
            continue
        full_path = os.path.join(file_path, f)
        if os.path.isfile(full_path) and f.lower().endswith(".mat"):
            data_path.append(full_path)
    return sorted(data_path)


def window_slice(data, time_steps):
    data = np.asarray(data).reshape(-1, 310)
    if data.shape[0] < time_steps:
        return np.empty((0, time_steps, 310), dtype=np.float32)
    xs = []
    for i in range(data.shape[0] - time_steps + 1):
        xs.append(data[i:i + time_steps])
    return np.stack(xs).astype(np.float32)


def get_number_of_label_n_trial(dataset_name):
    label_seed4 = [[1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
                   [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
                   [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]]
    label_seed3 = [[2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
                   [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
                   [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]]
    if dataset_name == "seed3":
        return 15, 3, label_seed3
    if dataset_name == "seed4":
        return 24, 4, label_seed4
    raise ValueError("Unexpected dataset name: {}".format(dataset_name))


def get_trial_lengths(dataset_name):
    if dataset_name == "seed3":
        return SEED_VIDEO_TIME
    if dataset_name == "seed4":
        return SEEDIV_VIDEO_TIME
    raise ValueError("Unexpected dataset name: {}".format(dataset_name))


def _flatten_label(label_array):
    label_array = np.asarray(label_array).reshape(-1)
    if label_array.size == 0:
        raise ValueError("Empty label array")
    return label_array.astype(np.int64)


def _extract_from_struct(sample, struct_key):
    if struct_key not in sample:
        return None, None
    struct_obj = sample[struct_key]
    if isinstance(struct_obj, np.ndarray) and struct_obj.dtype == np.object_ and struct_obj.size == 1:
        struct_obj = struct_obj.item()
    feature = getattr(struct_obj, "feature", None)
    label = getattr(struct_obj, "label", None)
    if feature is None or label is None:
        return None, None
    return feature, label


def _extract_feature_and_label(sample, session):
    struct_key = "dataset_session{}".format(session)
    feature, label = _extract_from_struct(sample, struct_key)
    if feature is not None and label is not None:
        return feature, label

    feature_key = struct_key + ".feature"
    label_key = struct_key + ".label"
    if feature_key in sample and label_key in sample:
        return sample[feature_key], sample[label_key]

    suffix_feature = ".feature"
    suffix_label = ".label"
    dynamic_feature_key = next((key for key in sample.keys() if key.endswith(suffix_feature)), None)
    dynamic_label_key = next((key for key in sample.keys() if key.endswith(suffix_label)), None)
    if dynamic_feature_key and dynamic_label_key:
        return sample[dynamic_feature_key], sample[dynamic_label_key]

    raise KeyError("Cannot find dataset_session{}.feature / label in mat file".format(session))


def _map_frame_labels(frame_labels, dataset_name):
    frame_labels = _flatten_label(frame_labels)
    if dataset_name == "seed3":
        unique_values = set(frame_labels.tolist())
        if unique_values.issubset({-1, 0, 1}):
            frame_labels = frame_labels + 1
    return frame_labels.astype(np.int64)


def _split_trials(feature, frame_labels, trial_lengths, expected_trial_labels, args, file_path):
    feature = np.asarray(feature)
    if feature.ndim != 2:
        feature = feature.reshape(feature.shape[0], -1)
    if feature.shape[1] != 310:
        raise ValueError("{} feature dim should be 310, got {}".format(file_path, feature.shape[1]))

    total_frames = sum(trial_lengths)
    if feature.shape[0] != total_frames:
        raise ValueError("{} total frame count mismatch: expected {}, got {}".format(
            file_path, total_frames, feature.shape[0]
        ))
    if frame_labels.shape[0] != total_frames:
        raise ValueError("{} label frame count mismatch: expected {}, got {}".format(
            file_path, total_frames, frame_labels.shape[0]
        ))

    trial_samples = []
    trial_labels = []
    start = 0
    for trial_index, trial_len in enumerate(trial_lengths):
        end = start + trial_len
        trial_feature = feature[start:end].reshape(trial_len, 310)
        trial_frame_labels = frame_labels[start:end]
        unique_labels = np.unique(trial_frame_labels)
        if unique_labels.size != 1:
            raise ValueError("{} trial {} contains non-constant frame labels: {}".format(
                file_path, trial_index, unique_labels.tolist()
            ))
        expected_label = int(expected_trial_labels[trial_index])
        actual_label = int(unique_labels[0])
        if actual_label != expected_label:
            raise ValueError("{} trial {} label mismatch: expected {}, got {}".format(
                file_path, trial_index, expected_label, actual_label
            ))
        windows = window_slice(trial_feature, args.time_steps)
        if windows.shape[0] > 0:
            trial_samples.append(windows)
            trial_labels.append(np.full((windows.shape[0], 1), expected_label, dtype=np.int64))
        start = end

    if not trial_samples:
        raise ValueError("{} produced no valid sliding windows".format(file_path))
    return np.concatenate(trial_samples, axis=0), np.concatenate(trial_labels, axis=0)


def load_trained_data(samples_path_list, args):
    _, _, labels = get_number_of_label_n_trial(args.dataset_name)
    expected_trial_labels = np.asarray(labels[int(args.session) - 1], dtype=np.int64)
    trial_lengths = get_trial_lengths(args.dataset_name)

    x_train_all = []
    y_train_all = []
    for path in samples_path_list:
        sample = scio.loadmat(path, verify_compressed_data_integrity=False, squeeze_me=True, struct_as_record=False)
        feature, frame_labels = _extract_feature_and_label(sample, args.session)
        frame_labels = _map_frame_labels(frame_labels, args.dataset_name)
        x_subject, y_subject = _split_trials(
            feature, frame_labels, trial_lengths, expected_trial_labels, args, path
        )
        x_train_all.append(x_subject)
        y_train_all.append(y_subject)
    return x_train_all, y_train_all


def normalize(features, select_dim=0):
    features_min, _ = torch.min(features, dim=select_dim)
    features_max, _ = torch.max(features, dim=select_dim)
    features_min = features_min.unsqueeze(select_dim)
    features_max = features_max.unsqueeze(select_dim)
    denom = torch.clamp(features_max - features_min, min=1e-8)
    return (features - features_min) / denom


def load4train(samples_path_list, args):
    train_sample, train_label = load_trained_data(samples_path_list, args)
    sample_res = []
    label_res = []
    for subject_index in range(len(train_sample)):
        one_subject_samples = torch.from_numpy(train_sample[subject_index]).type(torch.FloatTensor)
        one_subject_labels = torch.from_numpy(train_label[subject_index]).type(torch.LongTensor)
        one_subject_samples = normalize(one_subject_samples)
        sample_res.append(one_subject_samples)
        label_res.append(one_subject_labels)
    return sample_res, label_res


def getDataLoaders(one_subject, args):
    path_list = get_data_path(args.path)
    if not path_list:
        raise FileNotFoundError("No .mat files found in {}".format(args.path))
    if one_subject >= len(path_list):
        raise IndexError("Subject index {} out of range for {} files".format(one_subject, len(path_list)))

    target_path = path_list[one_subject]
    source_path_list = [path for idx, path in enumerate(path_list) if idx != one_subject]
    target_path_list = [target_path]

    sources_sample, sources_label = load4train(source_path_list, args)
    targets_sample, targets_label = load4train(target_path_list, args)
    target_sample = targets_sample[0]
    target_label = targets_label[0]

    source_dsets = []
    for i in range(len(sources_sample)):
        source_dsets.append(torch.utils.data.TensorDataset(sources_sample[i], sources_label[i]))
    target_dset = torch.utils.data.TensorDataset(target_sample, target_label)

    source_loaders = []
    for j in range(len(source_dsets)):
        source_loaders.append(torch.utils.data.DataLoader(
            source_dsets[j],
            args.batch_size,
            shuffle=True,
            num_workers=args.num_workers_train,
            drop_last=True
        ))
    test_loader = torch.utils.data.DataLoader(
        target_dset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers_test,
        drop_last=True
    )
    return source_loaders, test_loader
