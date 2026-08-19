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

CONFIG = {
    'data_path': './dataset/preprocess/CCSHI',
    'task': 'CCSHI',
    'n_classes': 5,
    'subject_list': ['A01', 'A03', 'A05', 'A06', 'A07', 'A08', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25'],
    'learning_rate': 0.0001,
    'batch_size': 64,
    'epochs': 1500,
    'n_splits': 5,
    'weight_decay': 0,
    'seed': 42,
    'selection_metric': 'accuracy',
    'classes_list': ['一', '丨', '丿', '㇏', 'ㄥ'],
    'base_output_folder': './output',
    'model_name': 'DRDNet'
}


def dictToYaml(filePath, dictToWrite):
    with open(filePath, 'w', encoding='utf-8') as f:
        yaml.dump(dictToWrite, f, allow_unicode=True)
    f.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    timestamp_folder = str(time.strftime('%Y-%m-%d--%H-%M', time.localtime()))
    global_output_path = os.path.join(CONFIG['base_output_folder'], CONFIG['task'], CONFIG['model_name'], timestamp_folder)
    all_results = []

    for subject_id in CONFIG['subject_list']:
        print(f"Processing subject: {subject_id}")

        out_path = os.path.join(
            global_output_path,
            f"sub_{subject_id}"
        )

        if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)

        dictToYaml(os.path.join(out_path, 'config.yaml'), CONFIG)

        try:
            train_dataset, test_dataset = get_datasets(
                CONFIG['data_path'],
                subject_id
            )
        except FileNotFoundError:
            print(f"Missing data file for subject: {subject_id}")
            continue

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

        eval_model = EvalModel(trained_model)
        test_accuracy = eval_model.test_model(
            test_dataset,
            out_path=out_path
        )

        all_results.append({'subject': subject_id, 'accuracy': test_accuracy})

    if len(all_results) > 0:
        accuracies = [res['accuracy'] for res in all_results]
        avg_accuracy = sum(accuracies) / len(accuracies)

        summary_file_path = os.path.join(global_output_path, 'final_summary_report.txt')

        with open(summary_file_path, 'w', encoding='utf-8') as f:
            f.write(f"Model: {CONFIG['model_name']}\n")
            f.write(f"Time: {timestamp_folder}\n")
            f.write(f"Epochs: {CONFIG['epochs']} | Batch: {CONFIG['batch_size']} | LR: {CONFIG['learning_rate']}\n")
            f.write(f"--------------------------------------------------\n")
            f.write(f"{'Subject ID':<15} | {'Test Accuracy (%)':<20}\n")
            f.write(f"--------------------------------------------------\n")

            for res in all_results:
                f.write(f"{res['subject']:<15} | {res['accuracy']:.2f}%\n")

            f.write(f"--------------------------------------------------\n")
            f.write(f"{'AVERAGE':<15} | {avg_accuracy:.2f}%\n")
            f.write(f"==================================================\n")

        print(f"Average accuracy: {avg_accuracy:.2f}%")

    print("All processing finished")


if __name__ == "__main__":
    if not os.path.exists(CONFIG['base_output_folder']):
        os.makedirs(CONFIG['base_output_folder'])
    main()