import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, classification_report,
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
        report_informations = classification_report(
            self.metrics['epoch_labels'],
            self.metrics['epoch_predictions'],
            labels=list(range(self.num_classes)),
            target_names=self.classes,
            output_dict=True,
            zero_division=0
        )
        
        for class_idx in range(self.num_classes):
            self.metrics['per_class_precisions'][class_idx].append(
                report_informations[self.classes[class_idx]]['precision']
            )
        
        for class_idx in range(self.num_classes):
            ax4.plot(self.metrics['per_class_precisions'][class_idx], 
                    label=f'{self.classes[class_idx]}')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Precision (%)')
        ax4.set_title('Per-Class Precision')
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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(42, 14), gridspec_kw={'width_ratios': [1, 1.5]})
        
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