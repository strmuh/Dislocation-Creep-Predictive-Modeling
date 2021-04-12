import pandas as pd
import numpy as np
from matplotlib import pyplot as py
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import preprocessing
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


class DataPreProcessing:
    def __init__(self, expdata, name='creep_data'):
        self.expdata = pd.read_csv(expdata)
        self.name = name

    def splitdata(self):
        # Split Data into features and labels
        features = self.expdata.drop('Strain', axis=1)
        labels = self.expdata['Strain'] * 1000
        # Form train, test and validate datasets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
        # Write data to csv files
        X_train.to_csv('train_features_' + self.name + '.csv', index=False)
        X_val.to_csv('val_features_' + self.name + '.csv', index=False)
        X_test.to_csv('test_features_' + self.name + '.csv', index=False)

        y_train.to_csv('train_labels_' + self.name + '.csv', index=False)
        y_val.to_csv('val_labels_' + self.name + '.csv', index=False)
        y_test.to_csv('test_labels_' + self.name + '.csv', index=False)


if __name__ == '__main__':
    DataPreProcessing()
