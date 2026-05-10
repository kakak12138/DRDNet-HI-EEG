import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib import rcParams
import os
import time


class TrainModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def train_model(self, model_class, n_classes, dataset, learning_rate=0.001, batch_size=64, epochs=500, n_splits=5,
                    weight_decay=0.015, seed=42, selection_metric='accuracy', out_path=None):
        self.set_seed(seed)

        # --- 设置日志文件 ---
        log_write = None
        if out_path:
            log_file_path = os.path.join(out_path, 'log_results.txt')
            log_write = open(log_file_path, 'w')
            log_write.write(f"--- Training Log ---\n")
            log_write.write(f"Timestamp: {time.strftime('%Y-%m-%d--%H-%M', time.localtime())}\n\n")

        if dataset.tensors[1].is_cuda:
            y_labels_for_stratify = dataset.tensors[1].cpu().numpy()
        else:
            y_labels_for_stratify = dataset.tensors[1].numpy()

        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        all_fold_accuracies = []
        all_fold_losses = []

        best_model_state = None
        best_epoch = 0
        best_val_acc_at_best_loss = 0.0
        best_val_loss_at_best_acc = float('inf')

        if selection_metric == 'accuracy':
            overall_best_val_metric = 0.0
        elif selection_metric == 'loss':
            overall_best_val_metric = float('inf')
        else:
            raise ValueError("selection_metric must be 'accuracy' or 'loss'")

        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset, y_labels_for_stratify)):
            print(f"Fold {fold + 1}/{n_splits}")
            if log_write:
                log_write.write(f"\n--- Fold {fold + 1}/{n_splits} ---\n")

            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=False)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=False)

            model = model_class(n_classes).to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            best_val_accuracy_in_fold = 0.0
            best_val_loss_in_fold = float('inf')

            train_accuracies = []
            train_losses = []
            val_accuracies = []
            val_losses = []
            best_val_acc_for_print = 0.0

            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                epoch_loss = running_loss / len(train_loader.dataset)
                epoch_accuracy = correct / total
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_accuracy)

                # Validation cycle
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                val_loss = val_loss / len(val_loader.dataset)
                val_accuracy = correct / total
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)

                if val_accuracy > best_val_acc_for_print:
                    print(
                        f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy * 100:.2f}%")
                    best_val_acc_for_print = val_accuracy

                if selection_metric == 'accuracy':
                    if val_accuracy > best_val_accuracy_in_fold:
                        best_val_accuracy_in_fold = val_accuracy
                        best_val_loss_in_fold = val_loss
                elif selection_metric == 'loss':
                    if val_loss < best_val_loss_in_fold:
                        best_val_loss_in_fold = val_loss
                        best_val_accuracy_in_fold = val_accuracy

                current_val_metric = val_accuracy if selection_metric == 'accuracy' else val_loss
                update_model = False

                if selection_metric == 'accuracy':
                    if current_val_metric > overall_best_val_metric:
                        update_model = True
                    elif current_val_metric == overall_best_val_metric:
                        if val_loss < best_val_loss_at_best_acc:
                            update_model = True
                else:
                    if current_val_metric < overall_best_val_metric:
                        update_model = True

                if update_model:
                    overall_best_val_metric = current_val_metric
                    best_val_acc_at_best_loss = val_accuracy if selection_metric == 'loss' else best_val_acc_at_best_loss
                    best_val_loss_at_best_acc = val_loss if selection_metric == 'accuracy' else val_loss
                    best_epoch = epoch + 1
                    best_model_state = model.state_dict()

            all_fold_accuracies.append(best_val_accuracy_in_fold)
            all_fold_losses.append(best_val_loss_in_fold)

            print(f"Best Validation Accuracy for Fold {fold + 1}: {best_val_accuracy_in_fold * 100:.2f}%")
            print(f"Best Validation Loss for Fold {fold + 1}: {best_val_loss_in_fold:.4f}\n")
            if log_write:
                log_write.write(f"Fold Best Validation Accuracy: {best_val_accuracy_in_fold * 100:.2f}%\n")
                log_write.write(f"Fold Best Validation Loss: {best_val_loss_in_fold:.4f}\n")

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
            plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Fold {fold + 1} - Loss')
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy')
            plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title(f'Fold {fold + 1} - Accuracy')
            plt.legend()
            plt.grid(True)

            if out_path:
                plot_save_path = os.path.join(out_path, f'fold_{fold + 1}_metrics.png')
                plt.savefig(plot_save_path)

            plt.close()

        average_val_accuracy = sum(all_fold_accuracies) / n_splits
        average_val_loss = sum(all_fold_losses) / n_splits

        metric_label = "Validation Accuracy" if selection_metric == 'accuracy' else "Validation Loss"

        print("\n--- 5-Fold Cross-Validation Summary ---")
        if selection_metric == 'accuracy':
            print(f"Average {metric_label}: {average_val_accuracy * 100:.2f}%")
        else:
            print(f"Average {metric_label}: {average_val_loss:.4f}")

        print(f"Best {selection_metric} (Global) achieved at epoch {best_epoch}")
        if log_write:
            log_write.write(f"\n--- 5-Fold Cross-Validation Summary ---\n")
            log_write.write(f"Average Validation Accuracy: {average_val_accuracy * 100:.2f}%\n")
            log_write.write(f"Average Validation Loss: {average_val_loss:.4f}\n")
            log_write.write(f"Best {selection_metric} (overall) achieved at epoch {best_epoch}\n")

        if selection_metric == 'loss':
            print(f"Best Global Val Loss: {overall_best_val_metric:.4f} (Acc: {best_val_acc_at_best_loss * 100:.2f}%)")
            if log_write:
                log_write.write(
                    f"Best Global Val Loss: {overall_best_val_metric:.4f} (Acc: {best_val_acc_at_best_loss * 100:.2f}%)\n")
        else:
            print(f"Best Global Val Acc: {overall_best_val_metric * 100:.2f}% (Loss: {best_val_loss_at_best_acc:.4f})")
            if log_write:
                log_write.write(
                    f"Best Global Val Acc: {overall_best_val_metric * 100:.2f}% (Loss: {best_val_loss_at_best_acc:.4f})\n")

        final_model = model_class(n_classes).to(self.device)
        final_model.load_state_dict(best_model_state)

        if out_path:
            model_save_path = os.path.join(out_path, 'best_model.pth')
            torch.save(best_model_state, model_save_path)
            print(f"Best model state saved to {model_save_path}")

        if log_write:
            log_write.close()

        return final_model


class EvalModel():
    def __init__(self, model):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    def test_model(self, test_dataset, out_path=None):
        self.model.eval()
        correct = 0
        total = 0
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = (correct / total) * 100
        print("/------------------------------/")
        print(f"Test Accuracy: {accuracy:.2f}%")
        print("/------------------------------/")

        if out_path:
            log_file_path = os.path.join(out_path, 'log_results.txt')
            with open(log_file_path, 'a') as log_write:
                log_write.write(f"\n--- Final Test Set Evaluation ---\n")
                log_write.write(f"Test Accuracy: {accuracy:.2f}%\n")
                log_write.write(f"---------------------------------\n")

        return accuracy

    def plot_confusion_matrix(self, test_dataset, classes, out_path=None):
        # rcParams['font.family'] = 'Times New Roman' # 已注释，避免字体报错
        self.model.eval()
        y_pred = []
        y_true = []
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                y_pred.append(predicted.item())
                y_true.append(labels.item())

        cf_matrix = confusion_matrix(y_true, y_pred)

        row_sums = cf_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1e-10
        cf_matrix = cf_matrix.astype('float') / row_sums[:, np.newaxis]

        df_cm = pd.DataFrame(cf_matrix, index=classes, columns=classes)
        fig, ax = plt.subplots(figsize=(12, 10))
        cax = ax.imshow(cf_matrix, cmap="Blues", vmin=0, vmax=1.0)
        cbar = fig.colorbar(cax)
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)

        for i in range(len(classes)):
            for j in range(len(classes)):
                font_color = "white" if cf_matrix[i, j] >= 0.5 else "black"
                ax.text(j, i, f"{cf_matrix[i, j]:.3f}",
                        ha="center", va="center", color=font_color, fontsize=16, fontweight='bold')

        plt.title('Confusion Matrix', fontsize=24)
        plt.xticks(rotation=0, ha="right", fontsize=16)
        plt.yticks(fontsize=16)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)

        plt.xlabel('Predicted Labels', fontsize=18)
        plt.ylabel('True Labels', fontsize=18)
        cbar.ax.tick_params(labelsize=16)
        cbar.outline.set_visible(True)
        cbar.outline.set_linewidth(2)
        cbar.outline.set_edgecolor('black')
        ax.tick_params(axis='both', width=2)
        cbar.ax.tick_params(width=2)
        plt.tight_layout()

        if out_path:
            cm_save_path = os.path.join(out_path, 'confusion_matrix.png')
        else:
            cm_save_path = 'confusion_matrix_model.png'

        plt.savefig(cm_save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {cm_save_path}")
        plt.close()