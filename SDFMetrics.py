import matplotlib.pyplot as plt
import seaborn as sn
import torch

from torcheval.metrics import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryConfusionMatrix
from torchmetrics.classification import BinaryJaccardIndex


class Metrics:
    def __init__(self, device):
        # List of values over epochs for plotting
        self.accuracy_list = []
        self.precision_list = []
        self.recall_list = []
        self.f1_list = []
        self.mIOU_list = []
        self.confusion_matrix = torch.zeros((2, 2), device=device)
        # Raw numbers being computed on
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0
        self.mIOU = 0.0
        # Containers that do running computation of metrics
        self.accuracy_metric = BinaryAccuracy(device=device)
        self.precision_metric = BinaryPrecision(device=device)
        self.recall_metric = BinaryRecall(device=device)
        self.f1_metric = BinaryF1Score(device=device)
        self.mIOU_metric = BinaryJaccardIndex().to(device)
        self.confusion_metric = BinaryConfusionMatrix(device=device, normalize='pred')

    def update(self, prediction, label):
        self.accuracy_metric.update(prediction, label)
        self.precision_metric.update(prediction, label)
        self.recall_metric.update(prediction, label)
        self.f1_metric.update(prediction, label)
        self.mIOU_metric.update(prediction, label)
        self.confusion_metric.update(prediction, label)

    def compute(self):
        self.accuracy = self.accuracy_metric.compute().item()
        self.precision = self.precision_metric.compute().item()
        self.recall = self.recall_metric.compute().item()
        self.f1_score = self.f1_metric.compute().item()
        self.mIOU = self.mIOU_metric.compute().item()
        self.confusion_matrix = self.confusion_metric.compute()

    def append(self):
        self.accuracy_list.append(self.accuracy)
        self.precision_list.append(self.precision)
        self.recall_list.append(self.recall)
        self.f1_list.append(self.f1_score)
        self.mIOU_list.append(self.mIOU)

    def plot(self, pred_folder, date):
        plt.plot(self.accuracy_list, color='cyan', label='Accuracy')
        plt.plot(self.mIOU_list, color='orange', label='mIOU')
        plt.plot(self.precision_list, color='green', label='Precision')
        plt.plot(self.recall_list, color='blue', label='Recall')
        plt.plot(self.f1_list, color='red', label='F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('Metric')
        plt.title('Metrics over epochs')
        plt.legend()
        plt.savefig(f'{pred_folder}{date}_metrics.png')
        plt.show()

        sn.heatmap(self.confusion_matrix.numpy(force=True), square=True, xticklabels='01', yticklabels='01', annot=True)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.savefig(f'{pred_folder}{date}_confusion.png')
        plt.show()

    def reset_metrics(self):
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0
        self.mIOU = 0.0
        self.accuracy_metric.reset()
        self.precision_metric.reset()
        self.recall_metric.reset()
        self.f1_metric.reset()
        self.mIOU_metric.reset()
        self.confusion_metric.reset()

    def reset_lists(self):
        self.accuracy_list.clear()
        self.precision_list.clear()
        self.recall_list.clear()
        self.f1_list.clear()
        self.mIOU_list.clear()
        self.confusion_matrix = torch.zeros((2, 2))
