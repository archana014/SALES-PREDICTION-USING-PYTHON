# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the Dataset
# Replace 'sales_data.csv' with the path to your dataset
df = pd.read_csv('/content/advertising.csv')

# Step 2: Explore the Data (Optional)
# Check the structure of the dataset
print(df.info())
print(df.describe())

# Visualize correlations (Optional)
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Step 3: Data Preprocessing
# Handle missing values, if any
df.fillna(df.mean(), inplace=True)

# Separate features (X) and the target variable (y)
X = df.drop('Sales', axis=1)  # Assuming 'Sales' is the target variable
y = df['Sales']

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Feature Scaling (Optional)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train the Model (Using Linear Regression here)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 7: Make Predictions
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Step 8: Evaluate the Model
# Calculate Mean Squared Error and R-squared for both training and test data
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f"Training MSE: {mse_train:.2f}, R-squared: {r2_train:.2f}")
print(f"Testing MSE: {mse_test:.2f}, R-squared: {r2_test:.2f}")

# Step 9: Plot Results (Optional)
plt.scatter(y_test, y_pred_test)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# Step 10: Interpret Results
# The model is now trained and evaluated.
# R-squared tells how well the model explains the variance in sales,
# and the MSE tells how close the predictions are to the actual values.
