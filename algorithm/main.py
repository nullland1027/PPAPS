import os
import time
import numpy as np
import torch

from datasets import PlantDataSet, AnimalDataSet, DatasetDL
from ml_preds import RFPredictor, XGBoostPredictor, LGBMPredictor
from dl_preds import NNPredictor, NNModel


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
    # dataset_obj.normalize('l2')
    np.save(os.path.join(target_path, kind + '_no_pro.npy'), dataset_obj.get_data())
    np.save(os.path.join(target_path, kind + '_no_pro_label.npy'), dataset_obj.get_label())


def rf_adjust_params(rf_predictor):
    """

    @param rf_predictor:
    @return:
    """
    search = {
        'n_estimators': list(range(100, 500, 50)),
        'min_samples_split': list(range(5, 20)),
        'min_samples_leaf': list(range(3, 10)),
        'min_weight_fraction_leaf': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    }
    print(rf_predictor.search_params(search))  # Predictor will update inner RF model automatically
    return rf_predictor


def xgboost_adjust_params(xgboost_predictor):
    """

    @param xgboost_predictor:
    @return:
    """
    search = {
        'max_depth': [4, 6, 8, 10],
        'n_estimators': [350, 400, 450],
        'min_child_weight': [4, 5, 6, 7],
        'gamma': [2, 5, 10],
        'subsample': [0.2, 0.5, 0.8],
        'colsample_bytree': [0.5, 0.6, 0.8],
        'scale_pos_weight': [0.2, 0.5, 0.9],
    }
    print(xgboost_predictor.search_params(search))
    return xgboost_predictor


def lgbm_adjust_params(lgb_predictor):
    """
    Output the best params
    @param lgb_predictor:
    @return:
    """
    search = {
        'num_leaves': [10, 20, 31, 40],
        'max_depth': [3, 5, 8, 10],
        'learning_rate': [0.1, 0.05],
        'n_estimators': list(range(100, 500, 100)),
        'subsample_for_bin': [100000, 200000, 500000],
        'min_split_gain': [0, 0.1, 0.5, 0.8],
        'min_child_weight': [0.001, 0.005, 0.1],
        'min_child_samples': [5, 10, 20, 50],
        'colsample_bytree': [0.2, 0.5, 0.8],
    }
    print(lgb_predictor.search_params(search))
    return lgb_predictor


def random_forest(kind, data, label, test_data):
    rf_params_bst = {  # best params
        'n_estimators': 200,
        'min_samples_split': 8,
        'min_samples_leaf': 3,
        'min_weight_fraction_leaf': 0.4,
        'n_jobs': -1
    }
    rf = RFPredictor(kind, np.load(data), np.load(label), **rf_params_bst)
    start_time = time.time()
    rf = rf_adjust_params(rf)
    end_time = time.time()
    print('Params searching costs', round((end_time - start_time) / 3600, 3), 'Hours')
    rf.train()
    rf.save_model('models')
    
    rf.load_model(os.path.join('models', 'random_forest_' + kind + '.pickle'))
    rf.predict(test_data)
    rf.output_metrix()


def xgboost(kind, data, label, test_data):
    xgb_params = {
        'n_estimators': 350,
        'max_depth': 4,
        'learning_rate': 0.1,

        'min_child_weight': 7,
        'gamma': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.5,
        'scale_pos_weight': 0.5,

        'objective': 'binary:logistic',  # loss function
        'eval_metric': 'auc',
        'n_jobs': -1,
        'nthread': -1,
        'random_state': 42
    }

    xgb = XGBoostPredictor(kind, np.load(data), np.load(label), **xgb_params)
    start_time = time.time()
    xgb = xgboost_adjust_params(xgb)
    end_time = time.time()
    print('Params searching costs', round((end_time - start_time) / 3600, 3), 'Hours')
    xgb.train()
    xgb.save_model('models')

    xgb.load_model(os.path.join('models', 'xgboost_' + kind + '.pickle'))
    xgb.predict(test_data)
    xgb.output_metrix()
    xgb.show_ROC_curve()


def lgbm(kind, data, label, test_data):
    lgbm_params = {
        'num_leaves': 10,
        'max_depth': 3,
        'learning_rate': 0.05,
        'n_estimators': 100,
        'subsample_for_bin': 100000,
        'min_split_gain': 0.5,
        'min_child_weight': 0.001,
        'min_child_samples': 50,
        'colsample_bytree': 0.5,

        'objective': 'binary',
        'metric': 'auc',  # 优化指标
        'n_jobs': -1,
        'force_col_wise': True,
        'random_state': 42
    }
    lgbm_predictor = LGBMPredictor(kind, np.load(data), np.load(label), **lgbm_params)
    start_time = time.time()
    lgbm_predictor = lgbm_adjust_params(lgbm_predictor)
    end_time = time.time()
    print('Params searching costs', round((end_time - start_time) / 3600, 3), 'Hours')
    lgbm_predictor.train()
    lgbm_predictor.save_model('models')

    lgbm_predictor.load_model('models/lgbm_' + kind + '.pickle')


if __name__ == '__main__':
    animal_data = os.path.join('raw_data', 'animal', 'animal.npy')  # (306, 1082)
    animal_label = os.path.join('raw_data', 'animal', 'animal_label.npy')
    plant_data = os.path.join('raw_data', 'plant', 'plant.npy')
    plant_label = os.path.join('raw_data', 'plant', 'plant_label.npy')


    # Deep NN
#     hyparams = {
#         'lr': 0.008,
#         'batch_size': 10,
#         'epoch': 100
#     }
#     pred = NNPredictor('animal', 1082, hyparams)
#     # pred.load_data(animal_data, animal_label, batch_size=hyparams['batch_size'])
#     # pred.train()
#     # pred.save_model()
    
#     pred.load_model('models/mlp_animal_e100.pth')
#     pred.load_blind_test('raw_data/animal/Blind_Animal.csv', 2)
#     pred.predict()

    
    # random_forest('plant', plant_data, plant_label, 'raw_data/plant/Blind_Plant.csv')