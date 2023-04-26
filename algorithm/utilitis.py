import os
import hashlib
import numpy as np
import pandas as pd

from sklearn import metrics
from matplotlib import pyplot as plt
from data_sets import PlantDataSet, AnimalDataSet
from ml_preds import RFPredictor, XGBoostPredictor, LGBMPredictor


def get_dataset_obj(path, kind: str):
    """
    @param path: file path or file object(Only for pandas.read_csv(file_obj) working)
    @param kind:
    @return: The `Animal` or `Plant` dataset object
    """
    dataset_obj = None
    if kind == 'plant':
        dataset_obj = PlantDataSet(path)
    elif kind == 'animal':
        dataset_obj = AnimalDataSet(path)
    else:
        raise Exception("No this item.")
        return
    return dataset_obj


def data_process(path, kind='plant'):
    """
    Do data process and save npy
    @param path: raw `.csv` file path
    @param kind: `plant` or `animal`
    @return: None
    """
    target_path = os.path.split(path)[0]

    dataset_obj = get_dataset_obj(path, kind)
    dataset_obj.data_clean()
    dataset_obj.normalize('l2')

    np.save(os.path.join(target_path, kind + '_no_pro.npy'), dataset_obj.get_data())
    np.save(os.path.join(target_path, kind + '_no_pro_label.npy'), dataset_obj.get_label())


def file_md5(file):
    """
    @param file: File object!!!
    @return:
    """
    md5 = hashlib.md5()  # md5 obj
    while True:  # 循环读取文件内容，并更新 MD5 对象
        data = file.read(4096)
        if not data:
            break
        md5.update(data)
    return md5.hexdigest()


def rf_pred(file_path, kind):
    """For web app"""
    data_frame = pd.read_csv(file_path)
    rf_predictor = RFPredictor(kind, None, None)
    rf_predictor.load_model(os.path.join("algorithm", "models", kind, "random_forest.pickle"))
    try:
        rf_predictor.predict(file_path)
        data_frame['prediction_res'] = pd.Series(list(rf_predictor.blind_y_pred))
        file_name = file_path[7:-4] + '_rf_' + kind + '.csv'
        data_frame.to_csv(os.path.join('downloads', file_name), index=False)
        return file_name
    except FileNotFoundError as e:
        return "FAILED"


def xgb_pred(file_path, kind):
    """For web app"""
    data_frame = pd.read_csv(file_path)
    xgb_predictor = RFPredictor(kind, None, None)
    xgb_predictor.load_model(os.path.join("algorithm", "models", kind, "xgboost.pickle"))
    try:
        xgb_predictor.predict(file_path)
        data_frame['prediction_res'] = pd.Series(list(xgb_predictor.blind_y_pred))
        file_name = file_path[7:-4] + '_xgb_' + kind + '.csv'
        data_frame.to_csv(os.path.join('downloads', file_name), index=False)
        return file_name
    except FileNotFoundError as e:
        return "FAILED"


def lgbm_pred(file_path, kind):
    """For web app"""
    data_frame = pd.read_csv(file_path)
    lgbm_predictor = RFPredictor(kind, None, None)
    lgbm_predictor.load_model(os.path.join("algorithm", "models", kind, "lgbm.pickle"))
    try:
        lgbm_predictor.predict(file_path)
        data_frame['prediction_res'] = pd.Series(list(lgbm_predictor.blind_y_pred))
        file_name = file_path[7:-4] + '_lgbm_' + kind + '.csv'
        data_frame.to_csv(os.path.join('downloads', file_name), index=False)
        return file_name
    except FileNotFoundError as e:
        return "FAILED"


def attention_pred():
    pass


def show_ROC_curve(blind_y_ture, blind_y_pred):
    """
    After the prediction to blind test data and Run to see the performance.
    @param blind_y_ture: true val
    @param blind_y_pred: prediction val
    @return: None
    """
    fpr, tpr, threshold = metrics.roc_curve(blind_y_ture, blind_y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.title('Validation ROC')
    plt.plot(fpr, tpr, 'b', label='Val AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
