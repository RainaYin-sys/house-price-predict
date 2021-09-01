# house-price-predict
## Introduction
The purpose of this project is to predict the final price of each home with With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa.

The description of data files as follows:
* train.csv - the training set
* test.csv - the test set
* sample_submission.csv - a benchmark submission from a linear regression on year and month of sale, lot square footage, and number of bedrooms

## Data preprocessing
* Drop the 'SalePrice' columns of training data and assign to y_train variable, then take the log value to Standardize data.
* Merge training, testing data together to facilitate data preprocessing
* One-hot-coding for content items in merged data
* Fill na with mean columns
* Standardize columns using the method of subtracting its mean and dividing by the variance
* Get the format of numpy array for training

## Build model and train
* use the mean of Ridge and RandomForestRegressor model predict to build an ensemble model
* The accuracy of model is up to 90%
* Save the predict result after taking expm1 value

# Getting started
python house_price_ensemble.py <path_to_input_dir> <path_to_output_dir>

# Acknowledgements
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview
