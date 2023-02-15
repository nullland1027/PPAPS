import os
import numpy as np
import torch

from datasets import PlantDataSet, AnimalDataSet, DatasetDL
from dl_preds import RFPredictor, XGBoostPredictor, LGBMPredictor, NNPredictor, NNModel
import time


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
        'n_estimators': list(range(100, 400, 50)),
        'min_samples_split': list(range(8, 20)),
        'min_samples_leaf': list(range(1, 5)),
        'min_weight_fraction_leaf': [0.3, 0.4, 0.5]
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


if __name__ == '__main__':
    animal_data = os.path.join('raw_data', 'animal', 'animal.npy')  # (306, 1082)
    animal_label = os.path.join('raw_data', 'animal', 'animal_label.npy')
    plant_data = os.path.join('raw_data', 'plant', 'plant.npy')
    plant_label = os.path.join('raw_data', 'plant', 'plant_label.npy')

    # data_process('raw_data/animal/animal_all_data.csv', kind='animal')
    # =========Random Forest ==============
    # rf_params_bst = {  # best params
    #     'n_estimators': 200,
    #     'min_samples_split': 8,
    #     'min_samples_leaf': 3,
    #     'min_weight_fraction_leaf': 0.4,
    #     'n_jobs': -1
    # }
    # rf = RFPredictor('animal', np.load(animal_data), np.load(animal_label), **rf_params_bst)
    # rf.load_model(os.path.join('models', 'random_forest_animal.pickle'))
    # rf.predict(os.path.join('raw_data', 'animal', 'Blind_Animal.csv'))
    # rf = rf_adjust_params(rf)
    # rf.train()

    # rf.output_metrix()

    # ========XGBoost ===================
    # xgb_params = {
    #     'n_estimators': 350,
    #     'max_depth': 4,
    #     'learning_rate': 0.1,
    #
    #     'min_child_weight': 7,
    #     'gamma': 10,
    #     'subsample': 0.8,
    #     'colsample_bytree': 0.5,
    #     'scale_pos_weight': 0.5,
    #
    #     'objective': 'binary:logistic',  # loss function
    #     'eval_metric': 'auc',
    #     'n_jobs': -1,
    #     'nthread': -1,
    #     'random_state': 42
    # }
    #
    # ani_xgb = XGBoostPredictor('animal', np.load(animal_data), np.load(animal_label), **xgb_params)
    # print(ani_xgb.search_params(xgb_search))  # output best score and params
    # ani_xgb.train()
    # ani_xgb.save_model('models')

    # ani_xgb.load_model(os.path.join('models', 'xgboost_animal.pickle'))
    # ani_xgb.predict(os.path.join('raw_data', 'animal', 'Blind_Animal.csv'))
    # ani_xgb.output_metrix()
    # ani_xgb.show_ROC_curve()

    # ============LGBM===============

    # lgbm_params = {
    #     'num_leaves': 10,
    #     'max_depth': 3,
    #     'learning_rate': 0.05,
    #     'n_estimators': 100,
    #     'subsample_for_bin': 100000,
    #     'min_split_gain': 0.5,
    #     'min_child_weight': 0.001,
    #     'min_child_samples': 50,
    #     'colsample_bytree': 0.5,
    #
    #     'objective': 'binary',
    #     'metric': 'auc',  # 优化指标
    #     'n_jobs': -1,
    #     'force_col_wise': True,
    #     'random_state': 42
    # }
    # lgbm_predictor = LGBMPredictor('animal', np.load(animal_data), np.load(animal_label), **lgbm_params)
    # start_time = time.time()
    # lgbm_predictor = lgbm_adjust_params(lgbm_predictor)
    # end_time = time.time()
    # print('Params searching costs', round((end_time - start_time) / 3600, 3), 'Hours')
    # lgbm_predictor.train()
    # lgbm_predictor.save_model('models')

    # Deep NN
    pred = NNPredictor('animal', 1082)
    pred.load_data(animal_data, animal_label, batch_size=2)

    # pred.train(30)
    # pred.save_model()
    print(pred.load_blind_test(csv_file='raw_data/animal/Blind_Animal.csv', batch_size=2))
