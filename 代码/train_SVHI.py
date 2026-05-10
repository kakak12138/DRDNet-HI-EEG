import warnings
warnings.filterwarnings("ignore")
import math
import copy
import random
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import DataLoader
import time
import torch.nn.functional as F
import yaml
from data_utils import get_datasets
from model.EEGNet import *
from model.TransNet import *
from model.ATCNet import *
from model.DRDNet import *
from model.MIFNet import *
from model.conformer import *
from model.train_eval_utils import TrainModel, EvalModel

# 'A01', 'A03', 'A05', 'A06', 'A07', 'A08', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25'
# accuracy, loss
# '一', '丨', '丿', '㇏', 'ㄥ'
# 'a', 'o', 'e', 'i', 'u', 'ü'
CONFIG = {
    'data_path': './dataset/preprocess/SVHI',  # 预处理后 .npy 文件的保存路径
    'task': 'SVHI',
    'n_classes': 6,
    'subject_list': ['A01', 'A03', 'A05', 'A06', 'A07', 'A08', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25'],
    'learning_rate': 0.0001,
    'batch_size': 64,
    'epochs': 1500,
    'n_splits': 5,
    'weight_decay': 0,
    'seed': 42,
    'selection_metric': 'accuracy',
    'classes_list': ['a', 'o', 'e', 'i', 'u', 'ü'],
    'base_output_folder': './output',
    'model_name': 'DRDNet'
}


def dictToYaml(filePath, dictToWrite):
    with open(filePath, 'w', encoding='utf-8') as f:
        yaml.dump(dictToWrite, f, allow_unicode=True)
    f.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. 创建唯一的输出时间戳 ---
    timestamp_folder = str(time.strftime('%Y-%m-%d--%H-%M', time.localtime()))

    # 【新增】定义总路径，方便后续保存总结文件
    global_output_path = os.path.join(CONFIG['base_output_folder'], CONFIG['task'], CONFIG['model_name'], timestamp_folder)

    # 【新增】用于存储所有被试结果的列表
    all_results = []

    for subject_id in CONFIG['subject_list']:

        print(f"\n" + "=" * 60)
        print(f"       开始处理被试: {subject_id}       ")
        print(f"       运行时间戳: {timestamp_folder}       ")
        print("=" * 60)

        # --- 2. 创建特定于被试的输出路径 ---
        out_path = os.path.join(
            global_output_path,  # 使用上面定义的 global_output_path
            f"sub_{subject_id}"
        )

        if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)

        # --- 3. 保存配置文件 ---
        dictToYaml(os.path.join(out_path, 'config.yaml'), CONFIG)

        # --- 4. 加载数据 ---
        try:
            train_dataset, test_dataset = get_datasets(
                CONFIG['data_path'],
                subject_id
            )
        except FileNotFoundError:
            print(f"\n" + "!" * 60)
            print(f"错误：未找到被试 {subject_id} 的 .npy 数据文件。")
            print(f"--- 跳过被试 {subject_id} ---")
            print("!" * 60 + "\n")
            continue

        # --- 5. 训练模型 ---
        print("\n--- 启动 k-Fold 交叉验证训练 ---")
        train_handler = TrainModel()
        trained_model = train_handler.train_model(
            model_class=globals()[CONFIG['model_name']],
            n_classes=CONFIG['n_classes'],
            dataset=train_dataset,
            learning_rate=CONFIG['learning_rate'],
            batch_size=CONFIG['batch_size'],
            epochs=CONFIG['epochs'],
            n_splits=CONFIG['n_splits'],
            weight_decay=CONFIG['weight_decay'],
            seed=CONFIG['seed'],
            selection_metric=CONFIG['selection_metric'],
            out_path=out_path
        )
        print(f"--- 被试 {subject_id} 训练完成 ---")

        # --- 6. 评估模型 ---
        print("\n--- 启动测试集评估 ---")
        eval_model = EvalModel(trained_model)

        # 测试模型准确率
        test_accuracy = eval_model.test_model(
            test_dataset,
            out_path=out_path
        )

        # 【新增】将当前被试的结果添加到列表中
        all_results.append({'subject': subject_id, 'accuracy': test_accuracy})

        print(f"--- 被试 {subject_id} 评估完成 ---")

    # --- 7. 【新增】所有循环结束后，生成汇总报告 ---
    if len(all_results) > 0:
        print(f"\n" + "=" * 60)
        print("       生成最终汇总报告...       ")

        # 计算平均准确率
        accuracies = [res['accuracy'] for res in all_results]
        avg_accuracy = sum(accuracies) / len(accuracies)

        # 定义汇总文件路径 (在 timestamp 文件夹下，与 sub_XXX 文件夹同级)
        summary_file_path = os.path.join(global_output_path, 'final_summary_report.txt')

        with open(summary_file_path, 'w', encoding='utf-8') as f:
            f.write(f"Model: {CONFIG['model_name']}\n")
            f.write(f"Time: {timestamp_folder}\n")
            f.write(f"Epochs: {CONFIG['epochs']} | Batch: {CONFIG['batch_size']} | LR: {CONFIG['learning_rate']}\n")
            f.write(f"--------------------------------------------------\n")
            f.write(f"{'Subject ID':<15} | {'Test Accuracy (%)':<20}\n")
            f.write(f"--------------------------------------------------\n")

            # 写入每个被试的成绩
            for res in all_results:
                f.write(f"{res['subject']:<15} | {res['accuracy']:.2f}%\n")

            f.write(f"--------------------------------------------------\n")
            f.write(f"{'AVERAGE':<15} | {avg_accuracy:.2f}%\n")
            f.write(f"==================================================\n")

        print(f"所有被试平均准确率: {avg_accuracy:.2f}%")

    print(f"\n" + "=" * 60)
    print("       所有被试处理完毕       ")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    if not os.path.exists(CONFIG['base_output_folder']):
        os.makedirs(CONFIG['base_output_folder'])
    main()