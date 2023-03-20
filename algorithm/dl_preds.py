import numpy as np
import os.path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from data_sets import AnimalDataSet, PlantDataSet, DatasetDL

import torch
import onnx
import onnxruntime as ort
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


class AttentionNet(nn.Module):
    def __init__(self, num_features: int, num_hidden: int, num_classes: int):
        super(AttentionNet, self).__init__()

        self.fc1 = nn.Linear(num_features, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_classes)

        # 定义注意力机制的参数
        self.attention_weights = nn.Parameter(torch.zeros(num_hidden, 1))
        self.attention_bias = nn.Parameter(torch.zeros(num_hidden))

    def forward(self, x):  # x 的维度为 (batch_size, num_features)
        # 计算隐藏层的输出
        h = F.relu(self.fc1(x))  # (batch_size, num_hidden)

        # 计算注意力权重
        weights = torch.matmul(h, self.attention_weights) + self.attention_bias
        weights = torch.tanh(weights)  # (batch_size, 1)
        weights = F.softmax(weights, dim=0)  # (batch_size, 1)

        # 加权平均隐藏层的输出，得到注意力向量
        # attention = torch.sum(weights * h, dim=0).resize((10, -1))  # (batch_size, num_hidden,)
        attention = weights * h  # (batch_size, num_hidden)
        # 计算分类输出
        out = self.fc2(attention)  # (batch_size, num_classes)
        return out


class NNPredictor:
    def __init__(self, kind: str, input_features: int, hyper_params):
        self._kind = kind  # `animal` or `plant`
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._epochs = hyper_params['epoch']
        self._batch_size = hyper_params['batch_size']
        self._bts_num = 0  # blind test sample number

        self.model = AttentionNet(input_features, 512, 2).to(self._device)  # Modify model here

        self._dataset = None
        self.dataloader_train = None
        self.dataloader_val = None  # Evaluation data to
        self.dataloader_blind_test = None  # Blind test data to see the final performance of predictor

        self._criteria = nn.CrossEntropyLoss()  # Loss function
        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=hyper_params['lr'])
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, 3, eta_min=0.00001, verbose=True)

    def load_data(self, data_filepath: str, label_filepath: str):
        self._dataset = DatasetDL(data_filepath, label_filepath)  # Create TensorDataset
        X_train, X_test, y_train, y_test = train_test_split(self._dataset.get_data(),
                                                            self._dataset.get_labels(),
                                                            test_size=0.3,
                                                            shuffle=True,
                                                            random_state=42)
        # Create Dataloader
        self.dataloader_train = DataLoader(TensorDataset(X_train.float(), y_train),
                                           batch_size=self._batch_size,
                                           drop_last=True,
                                           shuffle=True)
        self.dataloader_val = DataLoader(TensorDataset(X_test.float(), y_test),
                                         batch_size=self._batch_size,
                                         drop_last=True,
                                         shuffle=True)

    def load_blind_test(self, csv_file: str, batch_size: int):
        """
        Load blind test data and make data loader.
        @param csv_file: Blind test animal ro plant csv file
        @param batch_size: You know
        @return:
        """
        dataset = None
        if self._kind == 'animal':
            dataset = AnimalDataSet(csv_file)
        elif self._kind == 'plant':
            dataset = PlantDataSet(csv_file)
        else:
            raise Exception
        dataset.data_clean()  # Remove unwanted data
        dataset.normalize('l2')  # Data process
        dataset_dl = DatasetDL(npy_obj_data=dataset.get_data(), npy_obj_label=dataset.get_label())
        self._bts_num = len(dataset_dl)
        self.dataloader_blind_test = DataLoader(
            TensorDataset(dataset_dl.get_data().float(), dataset_dl.get_labels()),
            batch_size=batch_size,
            drop_last=False,
            shuffle=False
        )

    def check_gpu(self):
        print('device:', self._device)

    def train_loop(self):
        size = len(self.dataloader_train.dataset)  # the length of train data set
        train_loss = 0
        self.model.train()
        for batch, (X, y) in enumerate(self.dataloader_train):
            # forward
            X = X.to(self._device)
            y = y.to(self._device)
            outputs = self.model(X)  # predicted result
            loss = self._criteria(outputs, y)  # Compute the loss

            # Backpropagation

            self._optimizer.zero_grad()  # After getting rid of the gradients from the last round
            loss.backward()  # compute the gradients of all parameters we want the network to learn.
            self._optimizer.step()  # Update the model.
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
                X, y = X.to(self._device), y.to(self._device)
                output = self.model(X)
                pred_res = torch.argmax(output, 1)

                pred_labels.append(pred_res.cpu().data.numpy())
                true_labels.append(y.cpu().data.numpy())
                eval_loss += self._criteria(output, y).item() * X.size(0)
        eval_loss = eval_loss / size
        return {
            'Evaluation loss': eval_loss,
            'Prediction result': pred_labels,
            'True labels': true_labels,
        }

    def validate_loop(self):
        res_dict = self.__model_predict(self.dataloader_val)
        eval_loss = res_dict['Evaluation loss']
        true_labels = res_dict['True labels']
        pred_labels = res_dict['Prediction result']

        true_labels, pred_labels = np.concatenate(true_labels), np.concatenate(
            pred_labels)  # if batch_size != 1, this can put all res into one array
        acc = np.sum(true_labels == pred_labels) / len(pred_labels)
        print('Validation Loss: {:.6f}, Accuracy: {:6f}\n'.format(eval_loss, acc))

    def train(self):
        for t in range(self._epochs):
            print(f"Epoch {t + 1}-------------------------------")
            # print(self._optimizer.param_groups[0]['lr'])
            self.train_loop()
            self.validate_loop()
            self._optimizer.step()
            self._scheduler.step()  # Update learning rate
        print("Done!")

    def save_model(self):
        """Save the params of current model"""
        base_dir = os.path.abspath(os.path.dirname(__file__))
        name = 'attention_' + self._kind + '_e' + str(self._epochs) + '_b' + str(self._batch_size) + '.pth'
        model_path = os.path.join(base_dir, 'models', self._kind, name)
        if os.path.exists(model_path):  # Overwrite
            os.remove(model_path)
        torch.save(self.model.state_dict(), model_path)

    def save_model_onnx(self, pth_file, batch_size):
        """
        Must call save_model before this method!!!!!
        @param pth_file:
        @param batch_size:
        @return:
        """
        self.model.load_state_dict(torch.load(pth_file, map_location=self._device))
        self.model.eval()
        x = torch.randn(batch_size, 1, 1082, requires_grad=True)
        torch_out = self.model(x)

        # export
        torch.onnx.export(
            self.model,
            args=(x,),
            f='models/attention_' + self._kind + '.onnx',
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']

        )

    def load_model(self, model_file: str):
        """
        Can load the GPU model on cpu-only machine by map_location params
        @param model_file: id of file
        @return:
        """
        try:
            self.model.load_state_dict(torch.load(model_file, map_location=self._device))
        except FileNotFoundError as fnfe:
            print(fnfe, 'Load failed!')
            return False
        return True

    @staticmethod
    def load_model_onnx(onnx_file: str):
        try:
            onnx.checker.check_model(onnx_file)
        except onnx.checker.ValidationError as e:
            print("The model is invalid: %s" % e)
        else:
            print('The model is valid!')
            return ort.InferenceSession(onnx_file)

    def predict(self):
        """Do prediction on blind test data set"""
        res_dict = self.__model_predict(self.dataloader_blind_test)
        true_labels = res_dict['True labels']
        pred_labels = res_dict['Prediction result']

        true_labels, pred_labels = np.concatenate(true_labels), np.concatenate(pred_labels)
        acc = np.sum(true_labels == pred_labels) / len(pred_labels)
        print('Accuracy: {:6f}\n'.format(acc))
        print(classification_report(true_labels, pred_labels))
        return acc

    def infer(self):
        """Using onnx and do inference"""
        # Load onnx model
        sess = ort.InferenceSession("models/plant/attention_plant.onnx", providers=['CPUExecutionProvider'])
        pred_res = []
        # Prepare input data, Must be np.ndarray type

        for batch, (X, y) in enumerate(self.dataloader_blind_test):  # np.ndarray
            X = X.numpy()
            X = np.reshape(X, (3, 1, 1082))
            ort_inputs = {sess.get_inputs()[0].name: X}  # {'input': X}
            ort_outs = sess.run(None, ort_inputs)
            for i in range(len(ort_outs[0])):
                pred_res.append(np.argmax(ort_outs[0][i][0]))
        print(classification_report(np.concatenate((self.__model_predict(self.dataloader_blind_test)['True labels'])), pred_res))