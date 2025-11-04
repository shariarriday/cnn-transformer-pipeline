import matplotlib.pyplot as plt

class AdvancedMetricsTracker:
    def __init__(self, ):
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
        fig, (ax1) = plt.subplots(1, 1, figsize=(20, 10))
       
        # Plot losses
        ax1.plot(self.metrics['train_losses'], label='Train Loss')
        ax1.plot(self.metrics['val_losses'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()