import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing

unwanted_cols = ['index', 'gi_x', 'variation', 'UID', 'is_del', 'nutation', 'ancestor', 'pos_1', 'site']
label_cols = 'is_del'


class DataSet:
    """
    Super class of two data sets
    """

    def __init__(self, csv_path):
        """
        :param csv_path: raw data csv file path
        """
        self._cleaned_data = None
        self._len = 0
        self._X = None  # train data
        self._y = None  # train label
        self.df = pd.read_csv(csv_path)  # raw data no clean

    def data_clean(self):
        """
        Remove unwanted data item and convert the type into numpy.ndarray
        :return: Cleaned data without no-val features
        """
        self._y = np.array(self.df[label_cols])
        self._cleaned_data = self.df.drop(labels=unwanted_cols, axis=1)
        self._X = self._cleaned_data
        return self._cleaned_data

    def range_scale(self, low, up):
        """
        Scale the data into [low, up] range.
        @param low: lower bound
        @param up: upper bound
        @return: Dataframe object after scaling
        """
        scaler = preprocessing.MinMaxScaler(feature_range=(low, up))
        return scaler.fit_transform(self.df)

    def normalize(self, norm='l1'):
        """
        Ensure the data at the same order of magnitude to improve the comparability of data with different features.
        :param norm:
        :return: data after processing
        """
        self._X = preprocessing.normalize(self._cleaned_data, norm=norm)
        return self._X

    def get_data(self):
        """Return data that can be trained and validated without labels"""
        return self._X

    def get_label(self):
        """Return labels"""
        return self._y

    def __getitem__(self, item: int):
        """
        :param item: index
        :return: the item-th data and label
        """
        return self._X[item], self._y[item]

    def __len__(self):
        return self._len


class AnimalDataSet(DataSet):
    """Subclass of Animal data"""

    def __init__(self, csv_path):
        super().__init__(csv_path)


class PlantDataSet(DataSet):
    """Subclass of Plant data"""

    def __init__(self, csv_path):
        super().__init__(csv_path)


class DatasetDL(Dataset):
    """The specific Dataset for deep learning"""

    def __init__(self, data_filepath=None, label_filepath=None, npy_obj_data=None, npy_obj_label=None):
        self._X = None
        self._y = None
        if npy_obj_label is None and npy_obj_label is None:
            self._X = torch.from_numpy(np.load(data_filepath))
            self._y = torch.from_numpy(np.load(label_filepath))
        else:
            self._X = torch.from_numpy(npy_obj_data)
            self._y = torch.from_numpy(npy_obj_label)
        self._len = len(self._X)

    def change_label(self):
        """
        Change the label size from [1] into [2]
        examples: [0] -> [0, 1]; [1] -> [1, 0]
        When the model uses softmax classification and call this function
        @return None
        """
        new_tensor = torch.zeros(self._y.shape)
        for i in range(len(self._y)):
            if self._y[i][0].item() == 0:
                new_tensor[i][0] = 1
        self._y = torch.cat([self._y, new_tensor], 1)

    def get_data(self):
        return self._X

    def get_labels(self):
        return self._y

    def __getitem__(self, item):
        return self._X[item], self._y[item]

    def __len__(self):
        return self._len
