from sklearn import metrics
from matplotlib import pyplot as plt


def show_ROC_curve(blind_y_ture, blind_y_pred, title: str):
    """
    After the prediction to blind test data and Run to see the performance.
    @param blind_y_ture: true val
    @param blind_y_pred: prediction val
    @param title: title of diagram
    @return: None
    """
    fpr, tpr, threshold = metrics.roc_curve(blind_y_ture, blind_y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.plot(fpr, tpr, 'b', label='Val AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def show_ROC_curve_all(y_trues, y_preds, kind, title: str):
    """
    Draw a diagram in one picture
    @param y_trues: list of y_true
    @param y_preds: list of y_pred
    @param kind: `animal` or `plant`
    @param title:
    @return: None
    """
    fpr1, tpr1, _ = metrics.roc_curve(y_trues[0], y_preds[0])
    fpr2, tpr2, _ = metrics.roc_curve(y_trues[1], y_preds[1])
    fpr3, tpr3, _ = metrics.roc_curve(y_trues[2], y_preds[2])
    fpr4, tpr4, _ = metrics.roc_curve(y_trues[3], y_preds[3])
    roc_auc1: float = metrics.auc(fpr1, tpr1)
    roc_auc2: float = metrics.auc(fpr2, tpr2)
    roc_auc3: float = metrics.auc(fpr3, tpr3)
    roc_auc4: float = metrics.auc(fpr4, tpr4)
    plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.plot(fpr1, tpr1, label='Random Forest = %0.3f' % roc_auc1)
    plt.plot(fpr2, tpr2, label='XGBoost = %0.3f' % roc_auc2)
    plt.plot(fpr3, tpr3, label='LightGBM = %0.3f' % roc_auc3)
    plt.plot(fpr4, tpr4, label='Attention Neural Network = %0.3f' % roc_auc4)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
