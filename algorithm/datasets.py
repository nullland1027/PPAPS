import numpy as np
import pandas as pd
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
        Remove unwanted data item
        :return: Cleaned data without no-val features
        """
        self._y = self.df[label_cols]
        self._cleaned_data = self.df.drop(labels=unwanted_cols, axis=1)
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
