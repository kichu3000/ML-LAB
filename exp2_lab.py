
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("data/auto-mpg.csv")

print("First 5 rows:")
print(df.head())
print("\nDataset shape:")
print(df.shape)

# Preprocess the data
df.dropna(inplace=True)

X = df[["displacement"]]
y = df["mpg"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_linear_pred = linear_model.predict(X_test)

linear_mse = mean_squared_error(y_test, y_linear_pred)
linear_r2 = r2_score(y_test, y_linear_pred)

print("\n========== LINEAR REGRESSION ==========")
print("MSE :", linear_mse)
print("R2 Score :", linear_r2)

# Polynomial Regression
degrees = [2, 3, 4, 5]
results = []

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)

    y_poly_pred = poly_model.predict(X_test_poly)

    mse = mean_squared_error(y_test, y_poly_pred)
    r2 = r2_score(y_test, y_poly_pred)

    results.append([degree, mse, r2])

    print("\n========== POLYNOMIAL REGRESSION ==========")
    print("Degree :", degree)
    print("MSE :", mse)
    print("R2 Score :", r2)

# Model Comparison
print("\n========== MODEL COMPARISON ==========")
print("Linear Regression")
print("MSE :", linear_mse)
print("R2 Score :", linear_r2)

for result in results:
    print(
        "Polynomial Degree", result[0],
        "MSE :", result[1],
        "R2 Score :", result[2]
    )

# Visualization
for degree in degrees:
    poly = PolynomialFeatures(degree=degree)

    X_poly = poly.fit_transform(X)

    poly_model = LinearRegression()
    poly_model.fit(X_poly, y)

    X_plot = np.linspace(
        X["displacement"].min(),
        X["displacement"].max(),
        300
    ).reshape(-1, 1)

    X_plot_poly = poly.transform(X_plot)
    y_plot = poly_model.predict(X_plot_poly)

    plt.scatter(
        X,
        y,
        color="blue",
        label="Actual Data"
    )

    plt.plot(
        X_plot,
        y_plot,
        color="red",
        linewidth=2,
        label="Polynomial Regression"
    )

    plt.xlabel("Engine Displacement")
    plt.ylabel("Miles Per Gallon (MPG)")
    plt.title("Polynomial Regression: Displacement vs MPG")
    plt.legend()
    plt.grid()
    plt.show()
