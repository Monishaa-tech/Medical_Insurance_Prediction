# Import libraries
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load dataset
df = pd.read_csv("insurance.csv")

# Remove duplicates
df = df.drop_duplicates()

# Encode categorical data
df['sex'] = df['sex'].map({'male':1, 'female':0})
df['smoker'] = df['smoker'].map({'yes':1, 'no':0})
df['region'] = df['region'].map({
    'southwest':0,
    'southeast':1,
    'northwest':2,
    'northeast':3
})

# Features and Target
X = df.drop('charges', axis=1)
y = df['charges']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearRegression()

# Hyperparameters
param_grid = {
    'fit_intercept': [True, False],
    'positive': [True, False]
}

# Grid Search
grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring='r2'
)

grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Prediction
y_pred = best_model.predict(X_test)

# Evaluation
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# Save model
joblib.dump(best_model, "insurance_model.pkl")

print("Model Saved Successfully!")