import os
import time
import numpy as np
import torch
from algorithm import utility
from algorithm.data_sets import PlantDataSet, AnimalDataSet, DatasetDL
from algorithm.ml_preds import RFPredictor, XGBoostPredictor, LGBMPredictor
from algorithm.dl_preds import NNPredictor, AttentionNet
import argparse

parser = argparse.ArgumentParser(description='This is a demo script')
parser.add_argument('--kind', dest='kind', type=str, required=True,
                    help='please input `animal` or `plant`')
parser.add_argument('--algorithm', dest='algorithm', type=str,
                    help='4 types. `rf` for Random Forest, `xgboost` for XGBoost, `lgbm` for LightGBM, `att` for Attention Nerual Network')
parser.add_argument('--roc', action='store_true', required=False)
args = parser.parse_args()


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
    # colsample_bytree=0.5, learning_rate=0.05, max_depth=5, min_child_samples=10, min_child_weight=0.005, min_split_gain=0.5, n_estimators=200, num_leaves=10, subsample_for_bin=500000
    search = {
        'num_leaves': [64, 128],
        'max_depth': [5, 6],
        'learning_rate': [0.05],
        'n_estimators': [200, 300],
        'subsample_for_bin': [200000, 500000],
        'min_split_gain': [0.1, 0.5, 0.8],
        'min_child_weight': [0.005],
        'min_child_samples': [9, 10, 11],
        'colsample_bytree': [0.4, 0.5, 0.6],
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
    if data is None or label is None:
        rf = RFPredictor(kind, None, None, **rf_params_bst)
    else:
        rf = RFPredictor(kind, np.load(data), np.load(label), **rf_params_bst)
    # start_time = time.time()
    # rf = rf_adjust_params(rf)
    # end_time = time.time()
    # print('Params searching costs', round((end_time - start_time) / 3600, 3), 'Hours')
    # rf.train()
    # rf.save_model('algorithm/models')

    rf.load_model(os.path.join('algorithm', 'models', kind, 'random_forest.pickle'))
    return rf.predict(test_data)
    rf.output_metrix()
    rf.show_ROC_curve()


def xgboost(kind, data, label, test_data):
    """
    @param kind:
    @param data:
    @param label:
    @param test_data: CSV file
    @return:
    """
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
        'random_state': 42,
        'tree_method': 'hist'
    }
    if data is None or label is None:
        xgb = XGBoostPredictor(kind, None, None, **xgb_params)
    else:
        xgb = XGBoostPredictor(kind, np.load(data), np.load(label), **xgb_params)
    # start_time = time.time()
    # xgb = xgboost_adjust_params(xgb)  # turning params
    # end_time = time.time()
    # print('Params searching costs', round((end_time - start_time) / 3600, 3), 'Hours')
    # xgb.train()
    # xgb.save_model('algorithm/models')

    xgb.load_model(os.path.join('algorithm', 'models', kind, 'xgboost.pickle'))
    return xgb.predict(test_data)
    xgb.output_metrix()
    xgb.show_ROC_curve()


def lgbm(kind, data, label, test_data):
    lgbm_params = {
        'num_leaves': 10,
        'max_depth': 5,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'subsample_for_bin': 500000,
        'min_split_gain': 0.5,
        'min_child_weight': 0.005,
        'min_child_samples': 10,
        'colsample_bytree': 0.5,

        'objective': 'binary',
        'metric': 'auc',  # 优化指标
        'n_jobs': -1,
        'random_state': 42,
        #         'device': 'gpu'
    }
    if data is None or label is None:
        lgbm_predictor = LGBMPredictor(kind, None, None, **lgbm_params)
    else:
        lgbm_predictor = LGBMPredictor(kind, np.load(data), np.load(label), **lgbm_params)
    # start_time = time.time()
    # lgbm_predictor = lgbm_adjust_params(lgbm_predictor)
    # end_time = time.time()
    # print('Params searching costs', round((end_time - start_time) / 3600, 3), 'Hours')
    # lgbm_predictor.train()
    # lgbm_predictor.save_model('algorithm/models')

    lgbm_predictor.load_model(os.path.join('algorithm', 'models', kind, 'lgbm.pickle'))
    return lgbm_predictor.predict(test_data)
    lgbm_predictor.output_metrix()
    lgbm_predictor.show_ROC_curve()


def deep_learning(kind, data, label, test_data):
    hyparams = {
        'lr': 0.02,
        'batch_size': 8,
        'epoch': 80
    }
    attention_pred = NNPredictor(kind, 1082, hyparams)
    # attention_pred.load_data(data, label)
    attention_pred.load_blind_test(test_data, batch_size=3)
    # attention_pred.train()
    # attention_pred.save_model()
    attention_pred.load_model(os.path.join("algorithm", "models", kind, "attention.pth"))
    return attention_pred.predict()


def args_algorithm_choose(kind, data, label, btest):
    if args.algorithm.lower() == 'rf':
        random_forest(kind, data, label, btest)
    elif args.algorithm.lower() == 'xgboost':
        xgboost(kind, data, label, btest)
    elif args.algorithm.lower() == 'lgbm':
        lgbm(kind, data, label, btest)
    elif args.algorithm.lower() == 'att':
        deep_learning(kind, data, label, btest)
    else:
        raise argparse.ArgumentError(args.algorithm, 'algorithm must be one of `rf`, `xgboost`, `lgbm` and `att`!')


def draw_pics(kind, btest):
    res, y_trues, y_preds = [], [], []
    res.append(random_forest(kind, None, None, btest))
    res.append(xgboost(kind, None, None, btest))
    res.append(lgbm(kind, None, None, btest))
    res.append(deep_learning(kind, None, None, btest))
    for i in res:
        y_trues.append(i[0])
        y_preds.append(i[1])
    utility.show_ROC_curve_all(y_trues, y_preds, kind, title='ROC Curve of Plant Protein Prediction')


if __name__ == '__main__':
    animal_data = os.path.join('algorithm', 'raw_data', 'animal', 'animal.npy')  # (306, 1082)
    animal_label = os.path.join('algorithm', 'raw_data', 'animal', 'animal_label.npy')
    plant_data = os.path.join('algorithm', 'raw_data', 'plant', 'plant.npy')
    plant_label = os.path.join('algorithm', 'raw_data', 'plant', 'plant_label.npy')

    animal_btest = os.path.join("algorithm", "raw_data", "animal", "Blind_Animal.csv")
    plant_btest = os.path.join("algorithm", "raw_data", "plant", "Blind_Plant.csv")

    if args.kind.lower() == 'animal':
        if args.roc:
            draw_pics('animal', animal_btest)
        else:
            args_algorithm_choose('animal', animal_data, animal_label, animal_btest)
    elif args.kind.lower() == 'plant':
        if args.roc:
            draw_pics('plant', plant_btest)
        else:
            args_algorithm_choose('plant', plant_data, plant_label, plant_btest)
    else:
        raise argparse.ArgumentError(args.kind, 'kind must be one of `animal`, `plant` or `roc`!')
