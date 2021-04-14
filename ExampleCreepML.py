""" An example file for the data processing, Machine Learning model training,
and usage"""
from ML_Creep_Data_Preprocessing import DataPreProcessing
from ML_Model_Training import MLTrainingModel

a = DataPreProcessing('Creep_Data_Full.csv')
pdata = a.splitdata()
b = MLTrainingModel(pdata['train_features_'], pdata['train_labels_'],
                    pdata['val_features_'], pdata['val_labels_'], 'RFR', filecsv=False)
