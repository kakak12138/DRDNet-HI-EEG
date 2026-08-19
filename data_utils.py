import numpy as np
import os
import torch
from torch.utils.data import TensorDataset


def scale_data(data):
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    std[std == 0] = 1e-8
    return (data - mean) / std


def get_datasets(data_path, subject_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data_file = os.path.join(data_path, f'{subject_id}T_data.npy')
    train_label_file = os.path.join(data_path, f'{subject_id}T_label.npy')
    test_data_file = os.path.join(data_path, f'{subject_id}E_data.npy')
    test_label_file = os.path.join(data_path, f'{subject_id}E_label.npy')

    try:
        data_T = np.load(train_data_file)
        labels_T = np.load(train_label_file)
        data_E = np.load(test_data_file)
        labels_E = np.load(test_label_file)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        raise

    X_train_normalized = scale_data(data_T)
    X_test_normalized = scale_data(data_E)

    y_train_normalized = labels_T - np.min(labels_T)
    y_test_normalized = labels_E - np.min(labels_E)

    X_train = torch.Tensor(X_train_normalized).unsqueeze(1)
    y_train = torch.LongTensor(y_train_normalized)

    X_test = torch.Tensor(X_test_normalized).unsqueeze(1)
    y_test = torch.LongTensor(y_test_normalized)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    print(f"Subject {subject_id} Train: {X_train.shape} Test: {X_test.shape}")

    return train_dataset, test_dataset