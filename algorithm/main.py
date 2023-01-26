import os
import numpy as np
from datasets import PlantDataSet
from datasets import AnimalDataSet
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
    animal = os.path.join('raw_data', 'animal', 'animal_all_data.csv')
    plant = os.path.join('raw_data', 'plant', 'plant_all_data.csv')

    try:
        data_process(animal, 'animal')
    except Exception as e:
        print(e)
