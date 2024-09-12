import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from functools import partial

# Define a function to calculate distance to city center
def calculate_distance_to_city_center(lat, lon, city_center_lat=37.7749, city_center_lon=-122.4194):
    return np.sqrt((lat - city_center_lat) ** 2 + (lon - city_center_lon) ** 2)

# Load your dataset
df = pd.read_csv('house_prices.csv')

#Latitude and Longitude are misspelled in the dataset
# Rename the columns with incorrect names
df = df.rename(columns={'Lattitude': 'Latitude', 'Longtitude': 'Longitude'})

# Vectorize the calculation of distance to city center
calculate_distance_to_city_center_vectorized = np.vectorize(partial(calculate_distance_to_city_center, city_center_lat=37.7749, city_center_lon=-122.4194))

# Create new features(if we want for better performance)
df['Distance_to_City_Center'] = calculate_distance_to_city_center_vectorized(df['Latitude'], df['Longitude'])
df['Bedroom2_Squared'] = df['Bedroom2'] ** 2
df['BuildingArea_Squared'] = df['BuildingArea'] ** 2
df['Age_of_House'] = 2023 - df['YearBuilt']
df['Is_Near_Park'] = df['Distance'].apply(lambda x: 1 if x < 0.5 else 0)

# Drop categorical columns(columns with strings)
categorical_columns = ['Suburb', 'Address', 'Date', 'Type', 'Method', 'SellerG', 'CouncilArea', 'Regionname']
df = df.drop(categorical_columns, axis=1)

# Handle NaN values in the dataset(to remove errors)
df = df.dropna()  # Drop rows with NaN values

# Split your data into features (X) and target (y)
X = df.drop('Price', axis=1)
y = df['Price']

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean squared error: {mse:.2f}')

# Load the test data from a separate CSV file
test_df = pd.read_csv('test_data.csv')

#Latitude and Longitude are misspelled in the dataset
# Rename the columns with incorrect names
test_df = test_df.rename(columns={'Lattitude': 'Latitude', 'Longtitude': 'Longitude'})

# Vectorize the calculation of distance to city center (if necessary)
calculate_distance_to_city_center_vectorized = np.vectorize(partial(calculate_distance_to_city_center, city_center_lat=37.7749, city_center_lon=-122.4194))

# Create new features (if necessary)
test_df['Distance_to_City_Center'] = calculate_distance_to_city_center_vectorized(test_df['Latitude'], test_df['Longitude'])
test_df['Bedroom2_Squared'] = test_df['Bedroom2'] ** 2
test_df['BuildingArea_Squared'] = test_df['BuildingArea'] ** 2
test_df['Age_of_House'] = 2023 - test_df['YearBuilt']
test_df['Is_Near_Park'] = test_df['Distance'].apply(lambda x: 1 if x < 0.5 else 0)

# Drop categorical columns (string values containing columns)
categorical_columns = ['Suburb', 'Address', 'Date', 'Type', 'Method', 'SellerG', 'CouncilArea', 'Regionname']
test_df = test_df.drop(categorical_columns, axis=1)

# Drop the 'Price' column from the test data
test_df = test_df.drop('Price', axis=1)

# Handle NaN values in the test data (if necessary)
test_df = test_df.dropna()  # Drop rows with NaN values

# Prepare the test data for prediction
X_test = test_df

# Make predictions on the test data
predicted_prices = model.predict(X_test)

# Print the predicted prices along with their respective test data
for i, predicted_price in enumerate(predicted_prices):
    print(f"Predicted price: {predicted_price:.2f}")
    print("Respective test data:")
    print(test_df.iloc[i])
    print()
