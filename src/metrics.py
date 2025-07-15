import matplotlib.pyplot as plt

class AdvancedMetricsTracker:
    def __init__(self):
        self.reset()
       
    def reset(self):
        self.metrics = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
        }
   
    def update_epoch_metrics(self, train_loss, val_loss, lr):
        self.metrics['train_losses'].append(train_loss)
        self.metrics['val_losses'].append(val_loss)
        self.metrics['learning_rates'].append(lr)
   
    def plot_training_curves(self, save_path=None):
        """Plot comprehensive training curves"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(100, 50))
       
        # Plot losses
        ax1.plot(self.metrics['train_losses'], label='Train Loss')
        ax1.plot(self.metrics['val_losses'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
       
        # Plot learning rate
        ax2.plot(self.metrics['learning_rates'])
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True)
       
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')