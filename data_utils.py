import numpy as np
import os
import torch
from torch.utils.data import TensorDataset


def scale_data(data):
    # data shape: (N, Chans, Time)
    # 计算每个样本、每个通道的均值和标准差 (axis=-1 表示沿时间轴计算)
    # keepdims=True 让结果形状变为 (N, Chans, 1)，方便广播计算
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)

    # 防止除以 0 (如果某通道全是死线)
    std[std == 0] = 1e-8

    return (data - mean) / std


def get_datasets(data_path, subject_id):
    """
    加载预处理好的训练和测试数据，并进行归一化和TensorDataset创建。
    (此版本已简化和修正)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. 定义文件路径 ---
    train_data_file = os.path.join(data_path, f'{subject_id}T_data.npy')
    train_label_file = os.path.join(data_path, f'{subject_id}T_label.npy')
    test_data_file = os.path.join(data_path, f'{subject_id}E_data.npy')
    test_label_file = os.path.join(data_path, f'{subject_id}E_label.npy')

    # --- 2. 直接加载数据 (不再使用多余的列表和 i=0) ---
    try:
        data_T = np.load(train_data_file)
        labels_T = np.load(train_label_file)
        data_E = np.load(test_data_file)
        labels_E = np.load(test_label_file)
    except FileNotFoundError as e:
        print(f"错误：未找到Numpy数据文件。 {e}")
        print(f"请确保您已经为被试 {subject_id} 运行了 preprocess_ccshi.py")
        raise

    # --- 3. 归一化 (Z-Score) ---

    # 2. 归一化 (分别对训练集和测试集独立进行逐样本归一化)
    # 这种方法下，测试集使用自己的均值归一化是合法的，因为它不依赖其他样本
    X_train_normalized = scale_data(data_T)
    X_test_normalized = scale_data(data_E)

    # 3. 后续处理保持不变 (Labels 归一化等)
    y_train_normalized = labels_T - np.min(labels_T)
    y_test_normalized = labels_E - np.min(labels_E)

    # --- 4. 转换为 Tensors ---

    # 训练集
    X_train = torch.Tensor(X_train_normalized).unsqueeze(1)
    y_train = torch.LongTensor(y_train_normalized)

    # 测试集 (!! Bug修复：这里变量名从 y_train 改为 X_test !!)
    X_test = torch.Tensor(X_test_normalized).unsqueeze(1)
    y_test = torch.LongTensor(y_test_normalized)

    # --- 5. 创建 TensorDataset ---
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    print(f"Subject {subject_id} Train Dataset loaded: {X_train.shape}")
    print(f"Subject {subject_id} Test Dataset loaded: {X_test.shape}")

    return train_dataset, test_dataset