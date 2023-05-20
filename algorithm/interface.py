import os
import hashlib
import numpy as np
import pandas as pd

from algorithm.data_sets import PlantDataSet, AnimalDataSet
from algorithm.ml_preds import RFPredictor, XGBoostPredictor, LGBMPredictor
from algorithm.dl_preds import NNPredictor
from redis import Redis

r = Redis(host='127.0.0.1', port=6379, db=0)


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


def save_res(predictor):
    """
    Store the prediction result in databse
    @param predictor:
    @return: None
    """
    if r.exists('0') and r.exists('1'):
        r.set('0', int(r.get('0')) + list(predictor.blind_y_pred).count(0))
        r.set('1', int(r.get('0')) + list(predictor.blind_y_pred).count(1))
    else:
        r.set('0', list(predictor.blind_y_pred).count(0))
        r.set('1', list(predictor.blind_y_pred).count(1))


def rf_pred(file_path, kind):
    """For web app"""
    data_frame = pd.read_csv(file_path)
    rf_predictor = RFPredictor(kind, None, None)
    rf_predictor.load_model(os.path.join("algorithm", "models", kind, "random_forest.pickle"))
    try:
        rf_predictor.predict(file_path)
        save_res(rf_predictor)
        data_frame['prediction_result'] = pd.Series(list(rf_predictor.blind_y_pred))
        file_name = file_path[7:-4] + '_rf_' + kind + '.csv'
        data_frame.to_csv(os.path.join('downloads', file_name), index=False)
        return file_name
    except FileNotFoundError as e:
        return "FAILED"
    except Exception as base_error:
        return "FAILED"


def xgb_pred(file_path, kind):
    """For web app"""
    data_frame = pd.read_csv(file_path)
    xgb_predictor = XGBoostPredictor(kind, None, None)
    xgb_predictor.load_model(os.path.join("algorithm", "models", kind, "xgboost.pickle"))
    try:
        xgb_predictor.predict(file_path)
        save_res(xgb_predictor)
        data_frame['prediction_result'] = pd.Series(list(xgb_predictor.blind_y_pred))
        file_name = file_path[7:-4] + '_xgb_' + kind + '.csv'
        data_frame.to_csv(os.path.join('downloads', file_name), index=False)
        return file_name
    except FileNotFoundError as e:
        return "FAILED"
    except Exception as base_error:
        return "FAILED"


def lgbm_pred(file_path, kind):
    """For web app"""
    data_frame = pd.read_csv(file_path)
    lgbm_predictor = LGBMPredictor(kind, None, None)
    lgbm_predictor.load_model(os.path.join("algorithm", "models", kind, "lgbm.pickle"))
    try:
        lgbm_predictor.predict(file_path)
        save_res(lgbm_predictor)
        data_frame['prediction_result'] = pd.Series(list(lgbm_predictor.blind_y_pred))
        file_name = file_path[7:-4] + '_lgbm_' + kind + '.csv'
        data_frame.to_csv(os.path.join('downloads', file_name), index=False)
        return file_name
    except FileNotFoundError as e:
        return "FAILED"
    except Exception as base_error:
        return "FAILED"


def attention_pred(file_path, kind):
    """For web app"""
    att_predictor = NNPredictor(kind, 1082, {
        'lr': 0.02,
        'batch_size': 3,
        'epoch': 80
    })
    att_predictor.load_blind_test(file_path, 3)
    try:
        res = att_predictor.infer()
        if r.exists('0') and r.exists('1'):
            r.set('0', int(r.get('0')) + res.count(0))
            r.set('1', int(r.get('0')) + res.count(1))
        else:
            r.set('0', res.count(0))
            r.set('1', res.count(1))
        data_frame = pd.read_csv(file_path)
        data_frame['prediction_result'] = pd.Series(res)
        file_name = file_path[7:-4] + '_attention_' + kind + '.csv'
        data_frame.to_csv(os.path.join('downloads', file_name), index=False)
        return file_name
    except FileNotFoundError as e:
        return "FAILED"
    except Exception as base_error:
        return "FAILED"


class ObserverModel:
    def __init__(self):
        self.observers = []

    def add_observer(self, observer):
        self.observers.append(observer)

    def remove_observer(self, observer):
        self.observers.remove(observer)

    def notify_observers(self, message):
        for observer in self.observers:
            observer.update(message)


class Observer:
    def update(self, message):
        pass


class Logger(Observer):
    def update(self, message):
        print("Logging message:", message)
