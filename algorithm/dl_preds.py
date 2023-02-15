import numpy as np
import os.path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from datasets import AnimalDataSet, PlantDataSet, DatasetDL

import torch
import onnx
import onnxruntime
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


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
    def __init__(self, kind: str, input_features: int, hyper_params):
        self.kind = kind  # `animal` or `plant`
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = NNModel(input_features).to(self.device)
        self.dataset = None
        self.dataloader_train = None  #
        self.dataloader_val = None  # Evaluation data to
        self.dataloader_blind_test = None  # Blind test data to see the final performance of predictor
        self.criteria = nn.CrossEntropyLoss()  # Loss function
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hyper_params['lr'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 3)

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
        """
        Load blind test data and make data loader.
        @param csv_file: Blind test animal ro plant csv file
        @param batch_size: You know
        @return:
        """
        dataset = None
        if self.kind == 'animal':
            dataset = AnimalDataSet(csv_file)
        elif self.kind == 'plant':
            dataset = PlantDataSet(csv_file)
        else:
            raise Exception
        dataset.data_clean()  # Remove unwanted data
        dataset.normalize('l2')  # Data process
        dataset_dl = DatasetDL(npy_obj_data=dataset.get_data(), npy_obj_label=dataset.get_label())
        self.dataloader_blind_test = DataLoader(
            TensorDataset(dataset_dl.get_data().float(), dataset_dl.get_labels()),
            batch_size=batch_size,
            drop_last=True,
            shuffle=False
        )

    
    def check_gpu():
        print('device:', self.device)

    def train_loop(self):
        size = len(self.dataloader_train.dataset)  # the length of train data set
        train_loss = 0
        self.model.train()
        for batch, (X, y) in enumerate(self.dataloader_train):
            # forward
            X = X.to(self.device)
            y = y.to(self.device)
            outputs = self.model(X)  # predicted result
            loss = self.criteria(outputs, y)  # Compute the loss

            # Backpropagation

            self.optimizer.zero_grad()  # After getting rid of the gradients from the last round
            loss.backward()  # compute the gradients of all parameters we want the network to learn.
            self.optimizer.step()  # Update the model.
            train_loss += loss.item() * X.size(0)  # sum up the total loss
        train_loss = train_loss / size  # get average loss
        print('Training Loss: {:.6f}'.format(train_loss), end='        ')

    def __model_predict(self, loader):
        """Internal method, cannot be called by outside"""
        size = len(loader.dataset)
        eval_loss = 0
        true_labels, pred_labels = [], []  # the real label and the model prediction result
        self.model.eval()

        with torch.no_grad():  # evaluating
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                pred_res = torch.argmax(output, 1)
                
                pred_labels.append(pred_res.cpu().data.numpy())
                true_labels.append(y.cpu().data.numpy())
                eval_loss += self.criteria(output, y).item() * X.size(0)
        eval_loss = eval_loss / size
        return {
            'Evaluation loss': eval_loss,
            'Prediction result': pred_labels,
            'True labels': true_labels,
        }

    def evaluate_loop(self):
        res_dict = self.__model_predict(self.dataloader_val)
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
            print(self.optimizer.param_groups[0]['lr'])
            self.train_loop()
            self.evaluate_loop()
            self.optimizer.step()
            self.scheduler.step()
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

    def predict(self):
        """Do prediction on blind test data set"""
        pass
        res_dict = self.__model_predict(self.dataloader_blind_test)
        true_labels = res_dict['True labels']
        pred_labels = res_dict['Prediction result']

        true_labels, pred_labels = np.concatenate(true_labels), np.concatenate(pred_labels)
        acc = np.sum(true_labels == pred_labels) / len(pred_labels)
        print(pred_labels)
        print('Accuracy: {:6f}\n'.format(acc))
        print(classification_report(true_labels, pred_labels))



    def infer(self, npy_obj: np.ndarray):
        """Using onnx and do inference"""
        self.model.eval()
