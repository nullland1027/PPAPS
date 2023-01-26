import abc
import pickle
import xgboost
import numpy as np
from abc import ABC, abstractmethod
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics


class Predictor(ABC):
    """Abstract meta class"""
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def search_params(self, params_dict):
        pass

    @abstractmethod
    def save_model(self, name, path):
        pass

    @abstractmethod
    def output_metrix(self):
        pass


class RFPredictor(Predictor):
    def __init__(self, X, y, **kwargs):
        self._X = X
        self._y = y
        self.params = kwargs
        self.kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)  # Cross validation
        self.rfc = RandomForestClassifier(**self.params)

    def train(self):
        pass

    def search_params(self, params_dict):
        pass

    def save_model(self, name, path):
        pass


class XGBoostPredictor(Predictor):

    def train(self):
        pass

    def search_params(self, params_dict):
        pass

    def save_model(self, name, path):
        pass


class LGBMPredictor(Predictor):
    def train(self):
        pass

    def search_params(self, params_dict):
        pass

    def save_model(self, name, path):
        pass
