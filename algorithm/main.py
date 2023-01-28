import os
import numpy as np
from datasets import PlantDataSet
from datasets import AnimalDataSet
from networks import RFPredictor
from networks import XGBoostPredictor
import pandas as pd
from datasets import DataSet
from sklearn import preprocessing
from sklearn import model_selection


def data_process(path, kind='plant'):
    """
    Do data process and save npy
    @param path: raw data.csv file path
    @param kind: `plant` or `animal`
    @return: None
    """
    target_path = os.path.split(path)[0]
    dataset_obj = None
    if kind == 'plant':
        dataset_obj = PlantDataSet(path)
    elif kind == 'animal':
        dataset_obj = AnimalDataSet(path)
    else:
        raise Exception("No this item.")

    dataset_obj.data_clean()
    dataset_obj.normalize('l2')
    np.save(os.path.join(target_path, kind + '.npy'), dataset_obj.get_data())
    np.save(os.path.join(target_path, kind + '_label.npy'), dataset_obj.get_label())


def rf_adjust_params(rf_predictor):
    """

    @param rf_predictor:
    @return:
    """
    search = {
        'n_estimators': list(range(100, 400, 50)),
        'min_samples_split': list(range(8, 20)),
        'min_samples_leaf': list(range(1, 5)),
        'min_weight_fraction_leaf': [0.3, 0.4, 0.5]
    }
    print(rf_predictor.search_params(search))  # Predictor will update inner RF model automatically
    return rf_predictor


if __name__ == '__main__':
    animal_data = os.path.join('raw_data', 'animal', 'animal.npy')
    animal_label = os.path.join('raw_data', 'animal', 'animal_label.npy')
    plant_data = os.path.join('raw_data', 'plant', 'plant.npy')
    plant_label = os.path.join('raw_data', 'plant', 'plant_label.npy')

    # =========Random Forest ==============
    rf_params_bst = {  # best params
        'n_estimators': 250,
        'min_samples_split': 17,
        'min_samples_leaf': 4,
        'min_weight_fraction_leaf': 0.3,
        'n_jobs': -1
    }
    rf = RFPredictor('animal', np.load(animal_data), np.load(animal_label), **rf_params_bst)
    rf = rf_adjust_params(rf)
    rf.train()
    rf.save_model('models')

    # ========XGBoost ===================
    xgb_params = {
        'n_estimators': 400,
        'max_depth': 3,
        'learning_rate': 0.1,

        'min_child_weight': 5,
        'gamma': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.6,
        'scale_pos_weight': 1,

        'objective': 'binary:logistic',  # loss function
        'eval_metric': 'auc',
        'n_jobs': -1,
        'nthread': -1,
        'random_state': 42
    }

    xgb_search = {
        'max_depth': [4, 6, 8, 10],
        'n_estimators': [350, 400, 450],
        'min_child_weight': [4, 5, 6, 7],
        'gamma': [2, 5, 10],
        'subsample': [0.2, 0.5, 0.8],
        'colsample_bytree': [0.5, 0.6, 0.8],
        'scale_pos_weight': [0.2, 0.5, 0.9],
    }
    ani_xgb = XGBoostPredictor('animal', np.load(animal_data), np.load(animal_label), **xgb_params)
    print(ani_xgb.search_params(xgb_search))  # output best score and params
    ani_xgb.train()
    ani_xgb.save_model('/models')

    # ani_xgb.load_model(os.path.join('models', 'xgboost_animal.pickle'))
    # ani_xgb.predict(os.path.join('raw_data', 'animal', 'Blind_Animal.csv'))
    # print(ani_xgb.xgb)
    # ani_xgb.output_metrix()
    # ani_xgb.show_ROC_curve()
