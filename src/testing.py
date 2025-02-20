import torch
from sklearn.metrics import (classification_report)
from torch.optim.swa_utils import AveragedModel, update_bn
import pandas as pd
from tqdm import tqdm
import json

from .metrics import AdvancedMetricsTracker

def test_model(model, test_loader, device, num_classes, label_maps, label_reverse_maps, target):
    
    # Load best model
    checkpoint = torch.load('checkpoints/best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Initialize metrics tracker
    classes = []
    for i in range(num_classes):
        for item in label_reverse_maps[target].items():
            if item[1] == i:
                classes.append(label_maps[target][item[0]])

    metrics = AdvancedMetricsTracker(num_classes=num_classes, classes=classes)

    # Test loop
    correct = 0
    total = 0

    with torch.no_grad():
        for videos, labels in tqdm(test_loader, desc="Testing"):
            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update metrics
            metrics.update_predictions(predicted, labels, probabilities)

    # Calculate and display results
    test_accuracy = 100. * correct / total
    print(f'\nTest Accuracy: {test_accuracy:.2f}%')

    # Generate and save metrics plots
    metrics.plot_confusion_matrix(save_path='metrics/test_confusion_matrix.png')
    metrics.plot_roc_curves(save_path='metrics/test_roc_curves.png')
    metrics.plot_precision_recall_curves(save_path='metrics/test_pr_curves.png')

    # Generate classification report
    report = classification_report(
        metrics.metrics['epoch_labels'],
        metrics.metrics['epoch_predictions'],
        labels=list(range(num_classes)),
        target_names=metrics.classes,
        output_dict=True
    )

    # Save and display report
    with open('metrics/test_classification_report.json', 'w') as f:
        json.dump(report, f, indent=4)

    print('\nClassification Report:')
    print(pd.DataFrame(report).transpose())

    return test_accuracy, report