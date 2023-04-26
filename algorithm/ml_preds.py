import pickle
import os.path
import lightgbm as lgb
from abc import ABC, abstractmethod
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from data_sets import AnimalDataSet, PlantDataSet

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold


class Predictor(ABC):
    """Abstract meta class"""

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def search_params(self, params_dict: dict):
        pass

    @abstractmethod
    def save_model(self, path):
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

    @classmethod
    def _show_ROC_curve(cls, blind_y_ture, blind_y_pred):
        """
        After the prediction to blind test data and Run to see the performance.
        @param blind_y_ture: true val
        @param blind_y_pred: prediction val
        @return: None
        """
        fpr, tpr, threshold = metrics.roc_curve(blind_y_ture, blind_y_pred)
        roc_auc = metrics.auc(fpr, tpr)
        plt.figure(figsize=(6, 6))
        plt.title('Validation ROC')
        plt.plot(fpr, tpr, 'b', label='Val AUC = %0.3f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()


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
                               scoring='f1',
                               param_grid=params_dict,
                               n_jobs=-1,
                               verbose=10)  # using grid search to find the best params
        adapter.fit(self._X, self._y)
        self.rf_classifier = adapter.best_estimator_  # Update the model
        return adapter.best_score_, adapter.best_params_

    def save_model(self, path: str):
        """
        Save a model in the given path(folder)
        @param path: given folder
        @return: None
        """
        try:
            file_name = os.path.join(path, 'random_forest_' + self._kind) + '.pickle'
            if os.path.exists(file_name):
                os.remove(file_name)
            pickle.dump(self.rf_classifier, open(file_name, 'wb'))
            print('SAVED.')
        except FileExistsError as fee:
            print(fee)
        except FileNotFoundError as fnfe:
            print(fnfe)

    def load_model(self, file_name):
        self.rf_classifier = pickle.load(open(file_name, 'rb'))
        return self.rf_classifier

    def feature_importance(self):
        return self.rf_classifier.feature_importances_

    def predict(self, target_file):
        """

        @param target_file: csv file path (name)
        @return:
        """
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

    def show_ROC_curve(self):
        Predictor._show_ROC_curve(self.blind_y_ture, self.blind_y_pred)


class XGBoostPredictor(Predictor):
    def __init__(self, kind, X, y, **kwargs):
        self._kind = kind  # animal or plant
        self._X = X
        self._y = y
        self.params: dict = kwargs  # learner's params
        self._k_fold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)  # Cross validation
        self.xgb = XGBClassifier(**self.params)  # Build model
        self.blind_y_ture = None
        self.blind_y_pred = None

    def train(self):
        for train_idx, test_idx in self._k_fold.split(self._X, self._y):
            X_train, y_train = self._X[train_idx, :], self._y[train_idx]
            X_test, y_test = self._X[test_idx, :], self._y[test_idx]

            self.xgb.fit(X_train, y_train)
            print('Accuracy:', self.xgb.score(X_test, y_test))  # Show the accuracy of each epoch

    def search_params(self, params_dict):
        adapter = GridSearchCV(estimator=self.xgb,
                               scoring='roc_auc',
                               param_grid=params_dict,
                               n_jobs=-1,
                               cv=5,
                               verbose=10)  # using grid search to find the best params
        adapter.fit(self._X, self._y)
        self.xgb = adapter.best_estimator_  # Update the model=======THE MOST IMPORTANT CODE
        return adapter.best_score_, adapter.best_params_

    def feature_importance(self):
        return self.xgb.feature_importances_

    def save_model(self, path):
        """
        Save the XGBoost model
        @param path: The folder name NOT FILE NAME!!!!!!
        @return: None
        """
        try:
            file_name = os.path.join(path, 'xgboost_' + self._kind + '.pickle')
            if os.path.exists(file_name):
                os.remove(file_name)
            pickle.dump(self.xgb, open(file_name, 'wb'))
            print('MODEL SAVED.')
        except FileExistsError as fee:
            print('File exists ERROR!', fee)
        except FileNotFoundError as fnfe:
            print('File not found ERROR!', fnfe)

    def load_model(self, file_name):
        self.xgb = pickle.load(open(file_name, 'rb'))

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
        print(dataset_obj.get_data().shape)
        self.blind_y_pred = self.xgb.predict(dataset_obj.get_data())

    def output_metrix(self):
        Predictor._report(self.blind_y_ture, self.blind_y_pred)

    def show_ROC_curve(self):
        Predictor._show_ROC_curve(self.blind_y_ture, self.blind_y_pred)


class LGBMPredictor(Predictor):

    def __init__(self, kind, X, y, **kwargs):
        self._kind = kind  # animal or plant
        self._X = X
        self._y = y
        self.params: dict = kwargs  # learner's params
        self._k_fold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)  # Cross validation
        self.lgb_clf = LGBMClassifier(**self.params)  # Build model
        self.blind_y_ture = None
        self.blind_y_pred = None

    def search_params(self, params_dict):
        """

        @param params_dict:
        @return: best score and best params
        """
        adapter = GridSearchCV(estimator=self.lgb_clf,
                               scoring='roc_auc',
                               param_grid=params_dict,
                               cv=5,
                               n_jobs=-1,
                               verbose=10)  # using grid search to find the best params
        adapter.fit(self._X, self._y)
        self.lgb_clf = adapter.best_estimator_  # Update the model=======THE MOST IMPORTANT CODE
        return adapter.best_score_, adapter.best_params_

    def train(self):
        for train_idx, test_idx in self._k_fold.split(self._X, self._y):
            X_train, y_train = self._X[train_idx, :], self._y[train_idx]
            X_test, y_test = self._X[test_idx, :], self._y[test_idx]

            self.lgb_clf.fit(X_train, y_train)
            print('Accuracy:', self.lgb_clf.score(X_test, y_test))  # Show the accuracy of each epoch

    def save_model(self, path):
        try:
            file_name = os.path.join(path, self._kind, 'lgbm.pickle')
            if os.path.exists(file_name):
                os.remove(file_name)
            pickle.dump(self.lgb_clf, open(file_name, 'wb'))
            print('MODEL SAVED.')
        except FileExistsError as fee:
            print('File exists ERROR!', fee)
        except FileNotFoundError as fnfe:
            print('File not found ERROR!', fnfe)

    def load_model(self, file_name):
        self.lgb_clf = pickle.load(open(file_name, 'rb'))

    def output_metrix(self):
        Predictor._report(self.blind_y_ture, self.blind_y_pred)

    def feature_importance(self):
        """
        Draw the plot of feature importance
        @return:
        """
        return lgb.plot_importance(self.lgb_clf)

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
        self.blind_y_pred = self.lgb_clf.predict(dataset_obj.get_data())

    def show_ROC_curve(self):
        Predictor._show_ROC_curve(self.blind_y_ture, self.blind_y_pred)
