import abc
import os.path
from tqdm import tqdm
import pickle
import xgboost
import numpy as np
from abc import ABC, abstractmethod
from lightgbm import LGBMClassifier
from datasets import AnimalDataSet, PlantDataSet
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report


class Predictor(ABC):
    """Abstract meta class"""

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def search_params(self, params_dict: dict):
        pass

    @abstractmethod
    def save_model(self, name, path):
        pass

    @abstractmethod
    def load_model(self, file_name):
        pass

    @abstractmethod
    def output_metrix(self):
        """Output"""
        pass

    @abstractmethod
    def feature_importance(self):
        """Output feature importance"""
        pass

    @abstractmethod
    def predict(self, target_file):
        """
        Do infer
        @param target_file: csv format
        @return: The result of prediction
        """
        pass

    @classmethod
    def _report(cls, y_true, y_pred):
        """
        Output performance report.
        @param y_true: The real value
        @param y_pred: The prediction result
        @return: None
        """
        target_names = ['Class-0', 'Class-1']
        print(classification_report(y_true, y_pred, target_names=target_names))


class RFPredictor(Predictor):
    def __init__(self, kind, X, y, **kwargs):
        self._kind = kind  # animal or plant
        self._X = X
        self._y = y
        self.params: dict = kwargs  # learner's params
        self._k_fold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)  # Cross validation
        self.rf_classifier = RandomForestClassifier(**self.params)
        self.blind_y_ture = None
        self.blind_y_pred = None

    def train(self):
        for train_idx, test_idx in self._k_fold.split(self._X, self._y):
            X_train, y_train = self._X[train_idx, :], self._y[train_idx]
            X_test, y_test = self._X[test_idx, :], self._y[test_idx]

            self.rf_classifier.fit(X_train, y_train)
            print('Accuracy:', self.rf_classifier.score(X_test, y_test))

    def search_params(self, params_dict: dict):
        adapter = GridSearchCV(estimator=self.rf_classifier,
                               scoring='accuracy',
                               param_grid=params_dict,
                               verbose=10)  # using grid search to find the best params
        adapter.fit(self._X, self._y)
        self.rf_classifier = adapter.best_estimator_  # Update the model
        return adapter.best_score_, adapter.best_estimator_

    def save_model(self, name: str, path: str):
        pickle.dump(self.rf_classifier, open(os.path.join(path, name, '.pickle'), 'wb'))

    def load_model(self, file_name):
        return pickle.load(open(file_name, 'rb'))

    def feature_importance(self):
        return self.rf_classifier.feature_importances_

    def predict(self, target_file):
        dataset_obj = None
        if self._kind == 'plant':
            dataset_obj = PlantDataSet(target_file)
        elif self._kind == 'animal':
            dataset_obj = AnimalDataSet(target_file)
        else:
            raise Exception("No this item.")
        dataset_obj.data_clean()
        dataset_obj.normalize('l2')
        self.blind_y_ture = dataset_obj.get_label()
        self.blind_y_pred = self.rf_classifier.predict(dataset_obj.get_data())

    def output_metrix(self):
        Predictor._report(self.blind_y_ture, self.blind_y_pred)


class XGBoostPredictor(Predictor):
    def __init__(self):
        pass

    def load_model(self, file_name):
        pass

    def output_metrix(self):
        pass

    def feature_importance(self):
        pass

    def predict(self, target_file):
        pass

    def train(self):
        pass

    def search_params(self, params_dict):
        pass

    def save_model(self, name, path):
        pass


class LGBMPredictor(Predictor):
    def load_model(self, file_name):
        pass

    def output_metrix(self):
        pass

    def feature_importance(self):
        pass

    def predict(self, target_file):
        pass

    def train(self):
        pass

    def search_params(self, params_dict):
        pass

    def save_model(self, name, path):
        pass
