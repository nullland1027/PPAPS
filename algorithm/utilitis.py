import os
import hashlib
import numpy as np
import pandas as pd
from algorithm.data_sets import PlantDataSet, AnimalDataSet
from algorithm.ml_preds import RFPredictor, XGBoostPredictor, LGBMPredictor

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


def rf_pred(file, kind):
    dataset_obj = get_dataset_obj(file, kind)
    rf_predictor = RFPredictor(kind, None, None)
    rf_predictor.load_model(os.path.join("models", ""))


def xgb_pred():
    pass


def lgbm_pred():
    pass


def attention_pred():
    pass
