""" A script to train a Gradient Boosting Regression model (GBR) or Random Forest Regression (RFR)
 for Dislocation Creep
in 9-12% Cr Stainless Steels"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as py
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import warnings
import joblib

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


class MLTrainingModel:
    # Initialise Class properties
    def __init__(self, features, labels, val_features, val_labels, model='GBR', filecsv=True):
        self.features = features
        self.labels = labels
        self.val_features = val_features
        self.val_labels = val_labels
        self.model = model
        self.filecsv = filecsv

    # Import csv data into Dataframes
    def create_df(self):
        if self.filecsv:
            tr_features = pd.read_csv(self.features)
            tr_labels = pd.read_csv(self.labels, header=None)
            val_features = pd.read_csv(self.val_features)
            val_labels = pd.read_csv(self.val_labels, header=None)
        else:
            tr_features = pd.DataFrame(self.features)
            tr_labels = pd.DataFrame(self.labels)
            val_features = pd.DataFrame(self.val_features)
            val_labels = pd.DataFrame(self.val_labels)

        return tr_features, tr_labels, val_features, val_labels

    # Create GBR model based on experimental data
    def fit_model(self):
        # Perform Grid-Search for RFR model
        if self.model == 'RFR':
            gsc = GridSearchCV(
                estimator=RandomForestRegressor(),
                param_grid={
                    'max_depth': [2, 4, 8, 16, 32, None],
                    'n_estimators': [10, 50, 100, 1000],
                },
                cv=5, verbose=0, n_jobs=-1)
        else:
            # Perform Grid-Search for GBR model
            gsc = GridSearchCV(
                estimator=GradientBoostingRegressor(),
                param_grid={
                    'max_depth': [1, 3, 5, 7, 9],
                    'n_estimators': [5, 50, 250, 500],
                },
                cv=5, verbose=0, n_jobs=-1)
        # Fit model to Grid Parameters
        fitted_grid_result = gsc.fit(self.create_df()[0], self.create_df()[1].values.ravel())
        return gsc, fitted_grid_result

    # Create DataFrame for model performance results
    def full_results(self):
        grid_params = self.fit_model()[0]
        means = grid_params.cv_results_['mean_test_score']
        stds = grid_params.cv_results_['std_test_score']
        depth = []
        n_estimators = []
        for d in grid_params.cv_results_['params']:
            depth.append(d['max_depth'])
            n_estimators.append(d['n_estimators'])

        df_full_results = pd.DataFrame(list(zip(depth, n_estimators, means, stds)),
                                       columns=['Max_Depth', 'n_estimators', 'means', 'stds'])
        return df_full_results

    # Print fit results
    def print_results(self):
        fitted_results = self.fit_model()[1]
        print('BEST PARAMS: {}\n'.format(fitted_results.best_params_))
        means = fitted_results.cv_results_['mean_test_score']
        stds = fitted_results.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, fitted_results.cv_results_['params']):
            print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))

    # Pickle and save ML model
    def savemodel(self):
        joblib.dump(self.fit_model()[0].best_estimator_, self.model + '_model.pkl')

    def validate(self):
        R2Er = r2_score(self.create_df()[3], self.fit_model()[0].predict(self.create_df()[2]))
        RMSE = mean_squared_error(self.create_df()[3], self.fit_model()[0].predict(self.create_df()[2]), squared=False)
        print(RMSE, R2Er)

    # Plot model prediction results against experimental data
    def plot_results(self, stress, time, exp_data=None):
        test_time = np.arange(0, time, 1)
        stressplt = np.ones(time) * stress
        th_feat = pd.DataFrame({'Time': test_time, 'Stress': stressplt})

        py.plot(test_time, self.fit_model()[0].predict(th_feat), 'r.-', label=self.model + ' ML Prediction')
        py.xlabel('Time (hr)')
        py.ylabel('Stress (Pa)')
        py.legend(self.model + ' ML Prediction')
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
    MLTrainingModel()
