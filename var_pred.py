import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import scienceplots

# Read data from your local CSV file
# Replace 'your_file.csv' with your actual file path
df = pd.read_csv('C:/Users/Admin/PycharmProjects/2024/dataset/data.csv')

# Display the first few rows to verify data loading correctly
print("Data preview:")
print(df.head())

# Split features and target variable
# Make sure the column names match your CSV file
X = df[['X1', 'X2']]
y = df['Y']

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Get the model coefficients
intercept = model.intercept_
coef_X1 = model.coef_[0]
coef_X2 = model.coef_[1]

# Print the formula
print(f"\nMultiple Linear Regression Formula:")
print(f"Y = {intercept:.4f} + {coef_X1:.4f} * X1 + {coef_X2:.4f} * X2")

# Model evaluation metrics
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"\nModel Performance:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# Function to make predictions with new inputs
def predict_output(x1, x2):
    # Create a DataFrame with the same feature names used during training
    new_data = pd.DataFrame({'X1': [x1], 'X2': [x2]})
    return model.predict(new_data)[0]

# Example usage
new_x1 = 5  # replace with your new X1 value
new_x2 = 10  # replace with your new X2 value
predicted_y = predict_output(new_x1, new_x2)
print(f"\nPrediction Example:")
print(f"Predicted Y for X1={new_x1}, X2={new_x2}: {predicted_y:.4f}")

# # Visualize the actual vs predicted values
# plt.figure(figsize=(10, 6))
# plt.scatter(y, y_pred, color='blue', alpha=0.6)
# plt.plot([min(y), max(y)], [min(y), max(y)], 'r--')
# plt.xlabel('Actual Y')
# plt.ylabel('Predicted Y')
# plt.title('Actual vs Predicted Values')
# plt.savefig('prediction_performance.png')
# plt.show()

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Try polynomial features (degree 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Train model with polynomial features
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# Evaluate polynomial model
y_poly_pred = poly_model.predict(X_poly)
poly_r2 = r2_score(y, y_poly_pred)
poly_rmse = np.sqrt(mean_squared_error(y, y_poly_pred))

print("Polynomial Model Performance:")
print(f"R² Score: {poly_r2:.4f}")
print(f"RMSE: {poly_rmse:.4f}")

# Get feature names
feature_names = poly.get_feature_names_out(['X1', 'X2'])

# Get coefficients and intercept
coefficients = poly_model.coef_
intercept = poly_model.intercept_

# Better formula display
formula = f"Y = {intercept:.4f}"
for i, feature in enumerate(feature_names):
    coef = coefficients[i]

    # Replace with proper mathematical notation
    pretty_feature = feature.replace("^", "").replace(" ", "*")

    # Further clean up the notation
    pretty_feature = pretty_feature.replace("X1*X1", "X1²")
    pretty_feature = pretty_feature.replace("X2*X2", "X2²")
    pretty_feature = pretty_feature.replace("X1²*X1", "X1³")
    pretty_feature = pretty_feature.replace("X2²*X2", "X2³")

    if coef >= 0:
        formula += f" + {coef:.4f}*{pretty_feature}"
    else:
        formula += f" - {abs(coef):.4f}*{pretty_feature}"

print("\nDegree 3 Polynomial Regression Formula:")
print(formula)

# 应用 Science Plots 的 IEEE 样式（适合学术论文）
plt.style.use(['science', 'ieee'])  # 使用 IEEE 样式

# Visualize the actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y, y_poly_pred, s=50, marker="x", color="#d7191c", zorder=3)
plt.plot([min(y), max(y)], [min(y), max(y)], linestyle="--", linewidth=2, color="black")
plt.xlabel('Actual Y', fontsize = "14")
plt.ylabel('Predicted Y', fontsize = "14")
plt.xticks(fontsize = "14")
plt.yticks(fontsize = "14")
plt.savefig('poly_prediction_performance.png', dpi=600)
plt.show()




