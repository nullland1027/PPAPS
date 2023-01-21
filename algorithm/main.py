import os
import numpy as np
import pandas as pd
from datasets import DataSet
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier

data_path = os.path.join('raw_data', 'plant', 'Plant_train1.csv')
data_path2 = os.path.join('raw_data', 'plant', 'Plant_validation1.csv')

if __name__ == '__main__':
    train = pd.read_csv(data_path)
    val = pd.read_csv(data_path2)
    new = train.append(val)
    print(new.sort_values(by='index'))
