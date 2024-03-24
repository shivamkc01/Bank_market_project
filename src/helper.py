import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curve_for_classes(clf, x, y, class_labels, title):
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(class_labels):
        y_binary = (y == label).astype(int)
        y_score = clf.predict_proba(x)[:, i]
        fpr, tpr, _ = roc_curve(y_binary, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve - Class {label} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(f"../result_plots/roc_auc_curve/folds_without_sampling/{title}.jpg" ,dpi=100)
    plt.close()