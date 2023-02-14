import numpy as np
from tqdm import tqdm
import pickle
import os.path
from abc import ABC, abstractmethod
import lightgbm as lgb
import datetime
from copy import deepcopy
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from datasets import AnimalDataSet, PlantDataSet, DatasetDL

import torch
import onnx
import onnxruntime
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


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
            file_name = os.path.join(path, 'lgbm_' + self._kind + '.pickle')
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


class NNModel(nn.Module):

    def __init__(self, input_features: int):
        super(NNModel, self).__init__()
        # self.layer_1 = nn.Linear(input_features, 1024)
        # self.layer_2 = nn.Linear(1024, 64)
        # self.layer_out = nn.Linear(64, 1)
        #
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.1)
        # self.batch_norm1 = nn.BatchNorm1d(64)
        # self.batch_norm2 = nn.BatchNorm1d(64)

        self.stack = nn.Sequential(
            nn.Linear(input_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        y = self.stack(x)
        # x = self.relu(self.layer_1(x))
        # x = self.batch_norm1(x)
        # x = self.relu(self.layer_2(x))
        # x = self.batch_norm2(x)
        # x = self.dropout(x)
        # x = self.layer_out(x)
        return y


class NNPredictor:
    def __init__(self, kind: str, input_features: int):
        self.kind = kind  # `animal` or `plant`
        self.model = NNModel(input_features)
        self.dataset = None
        self.dataloader_train = None  #
        self.dataloader_val = None  # Evaluation data to
        self.dataloader_blind_test = None  # Blind test data to see the final performance of predictor
        self.criteria = nn.CrossEntropyLoss()  # Loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

    def load_data(self, data_filepath: str, label_filepath: str, batch_size: int):
        self.dataset = DatasetDL(data_filepath, label_filepath)  # Create TensorDataset
        # self.dataset.change_label()
        X_train, X_test, y_train, y_test = train_test_split(self.dataset.get_data(),
                                                            self.dataset.get_labels(),
                                                            test_size=0.3,
                                                            shuffle=True,
                                                            random_state=42)
        # Create Dataloader
        self.dataloader_train = DataLoader(TensorDataset(X_train.float(), y_train),
                                           batch_size=batch_size,
                                           drop_last=True,
                                           shuffle=True)
        self.dataloader_val = DataLoader(TensorDataset(X_test.float(), y_test),
                                         batch_size=batch_size,
                                         drop_last=True,
                                         shuffle=True)

    def load_blind_test(self, csv_file: str, batch_size):
        pass
        # TODO

    @staticmethod
    def check_gpu():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

    def train_loop(self):
        size = len(self.dataloader_train.dataset)  # the length of train data set
        train_loss = 0
        self.model.train()
        for batch, (X, y) in enumerate(self.dataloader_train):
            # forward
            outputs = self.model(X)  # predicted result
            loss = self.criteria(outputs, y)  # Compute the loss

            # Backpropagation
            self.optimizer.zero_grad()  # After getting rid of the gradients from the last round
            loss.backward()  # compute the gradients of all parameters we want the network to learn.
            self.optimizer.step()  # Update the model.
            train_loss += loss.item() * X.size(0)  # sum up the total loss
        train_loss = train_loss / size  # get average loss
        print('Training Loss: {:.6f}'.format(train_loss), end='        ')

    def model_predict(self):
        size = len(self.dataloader_val.dataset)
        eval_loss, acc = 0, 0
        true_labels, pred_labels = [], []  # the real label and the model prediction result
        self.model.eval()

        with torch.no_grad():  # evaluating
            for X, y in self.dataloader_val:
                output = self.model(X)
                pred_res = torch.argmax(output, 1)
                pred_labels.append(pred_res.data.numpy())
                true_labels.append(y.data.numpy())
                eval_loss += self.criteria(output, y).item() * X.size(0)
        eval_loss = eval_loss / size
        return {
            'Evaluation loss': eval_loss,
            'Prediction result': pred_labels,
            'True labels': true_labels,
        }

    def evaluate_loop(self):
        res_dict = self.model_predict()
        eval_loss = res_dict['Evaluation loss']
        true_labels = res_dict['True labels']
        pred_labels = res_dict['Prediction result']

        true_labels, pred_labels = np.concatenate(true_labels), np.concatenate(
            pred_labels)  # if batch_size != 1, this can put all res into one array
        acc = np.sum(true_labels == pred_labels) / len(pred_labels)
        print('Validation Loss: {:.6f}, Accuracy: {:6f}\n'.format(eval_loss, acc))

    def train(self, epoch_times):
        for t in range(epoch_times):
            print(f"Epoch {t + 1}-------------------------------")
            self.train_loop()
            self.evaluate_loop()
        print("Done!")

    def save_model(self):
        """Save the params of current model"""
        if os.path.exists('models/mlp_' + self.kind + '.pth'):
            os.remove('models/mlp_' + self.kind + '.pth')
        torch.save(self.model.state_dict(), 'models/mlp_' + self.kind + '.pth')

    def save_model_onnx(self, pth_file, batch_size):
        self.model.load_state_dict(pth_file)
        self.model.eval()
        x = torch.randn(batch_size, 1, 1082, requires_grad=True)
        torch_out = self.model(x)

        # export
        torch.onnx.export(
            self.model,
            x,
            'models/mlp_' + self.kind + '.onnx',
            export_params=True,
            opset_version=10,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']

        )

    def load_model(self, model_file):
        self.model.load_state_dict(torch.load(model_file))

    @staticmethod
    def load_model_onnx(onnx_file: str):
        try:
            onnx.checker.check_model(onnx_file)
        except onnx.checker.ValidationError as e:
            print("The model is invalid: %s" % e)
        else:
            print('The model is valid!')
            return onnxruntime.InferenceSession(onnx_file)

    def predict(self, npy_):
        pass
        # TODO

    def infer(self, npy_obj: np.ndarray):
        """Using onnx and do inference"""
        self.model.eval()
