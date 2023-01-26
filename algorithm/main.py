import os
import numpy as np
from datasets import PlantDataSet
from datasets import AnimalDataSet
from networks import RFPredictor
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


if __name__ == '__main__':
    animal_data = os.path.join('raw_data', 'animal', 'animal.npy')
    animal_label = os.path.join('raw_data', 'animal', 'animal_label.npy')
    # plant = os.path.join('raw_data', 'plant', 'plant_all_data.csv')

    rf_params_dft = {  # default params
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'min_weight_fraction_leaf': 0,
        'n_jobs': -1
    }
    rf = RFPredictor('animal', np.load(animal_data), np.load(animal_label), **rf_params_dft)
    # rf.train()
    search = {
        'n_estimators': list(range(50, 800, 50)),
        'max_depth': list(range(1, 20)),
        'min_samples_split': list(range(2, 10)),
        'min_samples_leaf': list(range(1, 20)),
        'min_weight_fraction_leaf': list(range(0, 10))
    }
    print(rf.search_params(search))
    rf.save_model('random_forest_animal', os.path.join('models'))
