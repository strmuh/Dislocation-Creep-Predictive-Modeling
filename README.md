## Table of Contents
1. [General Info](#general-info)
2. [Technologies](#technologies)
3. [Usage](#usage)

### General Info
***
Martensitic 9-12wt% Cr steels such as the designated P91 and P92 steels are used extensively 
as materials in thermal power plants. 
These steels are subjected to excessively high temperatures and stresses resulting in 
long-term material degradation of which creep deformation is often a chief component. As failure 
of critical components within power-plants can be catastrophic, a comprehensive understanding 
of the material failure due to creep is invaluable. The purpose of the current project
is to produce accurate prediction of long-term creep deformation using ML algorithms and techniques. 

## Technologies
***
A list of technologies used within the project:
* [Pandas](https://pandas.pydata.org/): Version 0.25.3 
* [Numpy](https://numpy.org/): Version 1.18.1
* [Matplotlib](https://matplotlib.org/): Version 3.1.2
* [Sklearn](https://scikit-learn.org/): Version 0.22.1
## Usage
***
This project currently consists of 3 python files:
1. ML_Creep_Data_Preprocessing.py
This file consists of a DataPreProcessing Class 
which provides methods for splitting raw experimental
creep-time data into Training, Validating and Testing
sets.

2. ML_Model_Training.py
This file consists of a MLTrainingModel Class 
which provides methods for training ML models using
processed data. ML models can be trained using either
a Gradient Boosting Regressor algorithm or Random Forrest 
Regressor algorithm. The MLTrainingModel Class has various
methods for model training, validating, saving and 
visualising results.  

3. ML_Model_Comparison.py
This file consists of a CompareModels Class 
which provides methods for comparing different ML models.
ML model performance is determined based on computation time,
accuracy and precision. Methods are also included for model visualisation

