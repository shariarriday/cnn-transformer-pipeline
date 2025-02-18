import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from collections import defaultdict

class AdvancedMetricsTracker:
    def __init__(self, num_classes, classes=None):
        self.num_classes = num_classes
        self.classes = classes if classes else [f"Class {i}" for i in range(num_classes)]
        self.reset()
        
    def reset(self):
        self.metrics = {
            'train_losses': [],
            'val_losses': [],
            'train_accuracies': [],
            'val_accuracies': [],
            'learning_rates': [],
            'per_class_accuracies': defaultdict(list),
            'per_class_precisions': defaultdict(list),
            'per_class_recalls': defaultdict(list),
            'per_class_f1_scores': defaultdict(list),
            'epoch_predictions': [],
            'epoch_labels': [],
            'epoch_probabilities': []
        }
    
    def calculate_per_class_metrics(self):
        """Calculate per-class accuracy, precision, recall and f1 scores"""
        y_true = np.array(self.metrics['epoch_labels'])
        y_pred = np.array(self.metrics['epoch_predictions'])
        
        for class_idx in range(self.num_classes):
            # Calculate binary metrics for current class
            true_binary = (y_true == class_idx)
            pred_binary = (y_pred == class_idx)
            
            # True Positives, False Positives, False Negatives
            tp = np.sum((true_binary) & (pred_binary))
            fp = np.sum((~true_binary) & (pred_binary))
            fn = np.sum((true_binary) & (~pred_binary))
            tn = np.sum((~true_binary) & (~pred_binary))
            
            # Calculate metrics
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Store metrics
            self.metrics['per_class_accuracies'][class_idx].append(accuracy * 100)
            self.metrics['per_class_precisions'][class_idx].append(precision * 100)
            self.metrics['per_class_recalls'][class_idx].append(recall * 100)
            self.metrics['per_class_f1_scores'][class_idx].append(f1 * 100)
    
    def update_epoch_metrics(self, train_loss, val_loss, train_acc, val_acc, lr):
        self.metrics['train_losses'].append(train_loss)
        self.metrics['val_losses'].append(val_loss)
        self.metrics['train_accuracies'].append(train_acc)
        self.metrics['val_accuracies'].append(val_acc)
        self.metrics['learning_rates'].append(lr)
    
    def update_predictions(self, predictions, labels, probabilities):
        self.metrics['epoch_predictions'].extend(predictions.cpu().numpy())
        self.metrics['epoch_labels'].extend(labels.cpu().numpy())
        self.metrics['epoch_probabilities'].extend(probabilities.cpu().numpy())
    
    def plot_training_curves(self, save_path=None):
        """Plot comprehensive training curves"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot losses
        ax1.plot(self.metrics['train_losses'], label='Train Loss')
        ax1.plot(self.metrics['val_losses'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(self.metrics['train_accuracies'], label='Train Acc')
        ax2.plot(self.metrics['val_accuracies'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Plot learning rate
        ax3.plot(self.metrics['learning_rates'])
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.grid(True)
        
        # Plot per-class metrics
        self.calculate_per_class_metrics()
        for class_idx in range(self.num_classes):
            ax4.plot(self.metrics['per_class_accuracies'][class_idx], 
                    label=f'{self.classes[class_idx]}')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy (%)')
        ax4.set_title('Per-Class Accuracy')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, save_path=None):
        """Plot enhanced confusion matrix"""
        cm = confusion_matrix(
            self.metrics['epoch_labels'],
            self.metrics['epoch_predictions'],
            labels=list(range(self.num_classes))
        )
        
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Raw counts
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d',
            cmap='Blues',
            xticklabels=self.classes,
            yticklabels=self.classes,
            ax=ax1
        )
        ax1.set_title('Confusion Matrix (Counts)')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        
        # Normalized
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.1%',
            cmap='Blues',
            xticklabels=self.classes,
            yticklabels=self.classes,
            ax=ax2
        )
        ax2.set_title('Confusion Matrix (Normalized)')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('True')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curves(self, save_path=None):
        """Plot ROC curves for each class"""
        plt.figure(figsize=(10, 8))
        
        # Compute ROC curve for each class
        for i in range(self.num_classes):
            # Prepare binary labels for current class
            y_true = np.array(self.metrics['epoch_labels']) == i
            y_score = np.array(self.metrics['epoch_probabilities'])[:, i]
            
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{self.classes[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    def plot_precision_recall_curves(self, save_path=None):
        """Plot precision-recall curves for each class"""
        plt.figure(figsize=(10, 8))
        
        for i in range(self.num_classes):
            y_true = np.array(self.metrics['epoch_labels']) == i
            y_score = np.array(self.metrics['epoch_probabilities'])[:, i]
            
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            avg_precision = average_precision_score(y_true, y_score)
            
            plt.plot(recall, precision, 
                    label=f'{self.classes[i]} (AP = {avg_precision:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()