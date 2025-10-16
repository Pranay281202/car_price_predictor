import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# 1. Load dataset
data = pd.read_csv('car_data.csv')  # Replace with your CSV path

# 2. Features and target
X = data[['year', 'km_driven', 'fuel', 'seller_type', 'transmission',
          'owner', 'mileage', 'engine', 'max_power', 'seats', 'brand']]
y = data['selling_price']

# 3. Separate numeric & categorical columns
numeric_features = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
categorical_features = ['fuel', 'seller_type', 'transmission', 'owner', 'brand']

# 4. Define preprocessing pipelines
numeric_transformer = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# 5. Full pipeline
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Fit model
pipe.fit(X_train, y_train)

# 8. Save pipeline
with open('pipe.pkl', 'wb') as f:
    pickle.dump(pipe, f)

print("✅ Model retrained and saved as pipe.pkl")
