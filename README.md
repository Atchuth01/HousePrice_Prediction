# HousePrice_Prediction
 Melbourne Houses prices prediction using Python [Machine Learning].


## House Price Prediction Model
* This repository contains a house price prediction model implemented in Python using scikit-learn and pandas. 
* The model takes in various features of a house, such as its location, size, and age, and predicts its price.

 ## Training data :
  * house_prices.csv

 ## Testing data :
  * test_data.csv


## Features
* The model uses a combination of numerical and categorical features to predict house prices.
* The features include:
* Location-based features (e.g. distance to city center, proximity to parks)
Property-based features (e.g. number of bedrooms, building area)
Temporal features (e.g. age of house)
* The model handles missing values and categorical variables using appropriate techniques.

## Model Training and Prediction
* The model is trained on a dataset of house prices and features.
* The trained model is used to make predictions on a separate test dataset.
* The predicted prices are printed in a vertical form along with their respective test data.

## Code Organization
* The code is organized into a single Python script (predict_house_prices.py) that loads the test data, preprocesses it, trains the model, makes predictions, and prints the results.
* The script uses pandas for data manipulation and scikit-learn for model training and prediction.

## Requirements
Python 3.x
pandas
scikit-learn
numpy

## Usage
* Clone the repository and navigate to the project directory.
* Install the required packages using pip install -r requirements.txt.
* Run the script using python predict_house_prices.py.
* The script will print the predicted prices along with their respective test data.

## License
* This project is licensed under the MIT License. See LICENSE for 