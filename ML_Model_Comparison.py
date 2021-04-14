""" A script to compare ML models used for 9-12% Cr Stainless Steels Creep
Predictions"""
from time import time
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
from matplotlib import pyplot as py
import numpy as np


class CompareModels:
    # Initialise Class properties
    def __init__(self, val_features, val_labels, mdlnames=('GBR', 'RFR')):
        self.val_features = pd.read_csv(val_features)
        self.val_labels = pd.read_csv(val_labels, header=None)
        self.mdlnames = mdlnames

    # Load Pickled models into a dictionary
    def load_models(self):
        models = {}
        for mdl in self.mdlnames:
            models[mdl] = joblib.load('{}_model.pkl'.format(mdl))
        return models

    def evaluate_model(self):
        models = self.load_models()
        for name, mdl in models.items():
            start = time()
            pred = mdl.predict(self.val_features)
            end = time()
            accuracy = round(r2_score(self.val_labels, pred), 3)
            RMSE = round(mean_absolute_error(self.val_labels, pred), 3)
            print('{} --R2 Accuracy: {} /RMSE Accuracy: {} / Computation time: {}'.format(name, accuracy, RMSE,
                                                                                          round(end - start, 3)))

    def plot_results(self, stress, time, exp_data=None):
        test_time = np.arange(0, time, 1)
        stressplt = np.ones(time) * stress
        th_feat = pd.DataFrame({'Time': test_time, 'Stress': stressplt})
        models = self.load_models()
        colors = {'GBR': 'r.-', 'RFR': 'y.-'}
        for name, mdl in models.items():
            py.plot(test_time, mdl.predict(th_feat), colors[name], label=name + ' ML Prediction')
            py.xlabel('Time (hr)')
            py.ylabel('Stress (Pa)')
            py.legend(name + ' ML Prediction')
            py.title('Dislocation Creep Plot')
        if exp_data is not None:
            expsub = 0
            test_no = 1
            for n, i in enumerate(exp_data['Time '][0:-1]):
                if exp_data['Time '][n + 1] < i:
                    py.plot(exp_data['Time '][expsub:n + 1], exp_data['Strain'][expsub:n + 1] * 1000, 'b.-',
                            label='Exp Test No' + str(test_no))
                    py.legend()
                    test_no += 1
                    expsub = n + 1
                elif len(exp_data['Time ']) == n + 2:
                    py.plot(exp_data['Time '][expsub:n + 2], exp_data['Strain'][expsub:n + 2] * 1000, 'b.-',
                            label='Exp Test No' + str(test_no))
        py.legend()


if __name__ == '__main__':
    CompareModels()


