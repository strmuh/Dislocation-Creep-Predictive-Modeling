""" An example file for the data processing, Machine Learning model training,
and usage"""
# Import required modules
from ML_Creep_Data_Preprocessing import DataPreProcessing
from ML_Model_Training import MLTrainingModel
from ML_Model_Comparison import CompareModels

# Initiate an instance of the "DataDataPreProcessing" class and input creep data from a csv file
data_processing = DataPreProcessing('Creep_Data_Full.csv')
# Use the "splitdata()" method to return the train-test-validate data as pandas DataFrames
pdata = data_processing.splitdata()

# Use the "savedata()" method to save the train-test-validate data as csv files in the working folder
# data_processing.savedata()

# Train a Random Forest Regression or Gradient Boosting Regression model using processed data.
# Create an instance of the "MLTrainingModel" with train and test features and labels as inputs
# Set filecsv=False if NOT loading CSV file data
# Set model type input to 'RFR' for Random Forest Regression or 'GBR'for Gradient Boosting Regression
RFR_model = MLTrainingModel(pdata['train_features_'], pdata['train_labels_'],
                            pdata['test_features_'], pdata['test_labels_'], 'RFR', filecsv=False)

GBR_model = MLTrainingModel(pdata['train_features_'], pdata['train_labels_'],
                            pdata['test_features_'], pdata['test_labels_'], 'GBR', filecsv=False)

# Save models as pkl files
RFR_model.savemodel()
GBR_model.savemodel()

# Compare RFR and GBR models by creating an instance of the CompareModels class
# Set filecsv=False if NOT loading CSV file data
compareMLmodels = CompareModels(pdata['val_features_'], pdata['val_labels_'], filecsv=False)
