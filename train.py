import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

print("1. Generating realistic dummy dataset for used cars...")
np.random.seed(42)
n_samples = 500

# Generating random features
brands = np.random.choice(['Toyota', 'Ford', 'BMW', 'Honda', 'Hyundai'], n_samples)
years = np.random.randint(2010, 2024, n_samples)
kilometers = np.random.randint(10000, 200000, n_samples)
damaged = np.random.choice(['No', 'Yes'], n_samples, p=[0.8, 0.2]) # 20% of cars are damaged

# Calculating a realistic price based on the features
# Base price + value for newer year - depreciation for km - massive drop for damage
prices = 20000 + ((years - 2010) * 1500) - (kilometers * 0.05)
# Apply damage penalty
prices = [p - 5000 if d == 'Yes' else p for p, d in zip(prices, damaged)]
# Add some random market noise
prices = np.array(prices) + np.random.normal(0, 1500, n_samples)

# Create DataFrame
df = pd.DataFrame({
    'brand': brands,
    'year': years,
    'kilometers': kilometers,
    'damaged': damaged,
    'price': prices
})

# Define Inputs (X) and Output (y)
X = df[['brand', 'year', 'kilometers', 'damaged']]
y = df['price']

print("2. Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("3. Building Machine Learning Pipeline...")
# We use OneHotEncoder for text data (brand, damaged) and StandardScaler for numbers (year, km)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['year', 'kilometers']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['brand', 'damaged'])
    ])

# Create a pipeline that preprocesses the data, then trains a Random Forest model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

print("4. Training the Model...")
model_pipeline.fit(X_train, y_train)

print("5. Evaluating the Model...")
predictions = model_pipeline.predict(X_test)
error = mean_absolute_error(y_test, predictions)
print(f"-> Model Error Margin: Off by roughly ${error:.2f} per prediction")

print("6. Saving the trained model to disk...")
joblib.dump(model_pipeline, 'car_price_model.pkl')
print("-> Success! Model saved as 'car_price_model.pkl'")