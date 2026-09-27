import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,r2_score

# Load dataset
df=pd.read_csv("data/auto-mpg.csv")
print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)
df.dropna(inplace=True)

# Select feature and target
X=df[["displacement"]]
y=df["mpg"]

# Split dataset
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)

# Linear Regression
linear_model=LinearRegression()
linear_model.fit(X_train,y_train)
y_linear_pred=linear_model.predict(X_test)

linear_mse=mean_squared_error(y_test,y_linear_pred)
linear_r2=r2_score(y_test,y_linear_pred)

print("\n--- Linear Regression Results ---")
print("MSE :",linear_mse)
print("R2 Score :",linear_r2)

# Polynomial Regression for different degrees
degrees=[2,3,4,5]
results=[]

for degree in degrees:

    # Create polynomial features
    poly=PolynomialFeatures(degree=degree)
    X_train_poly=poly.fit_transform(X_train)
    X_test_poly=poly.transform(X_test)

    # Train polynomial model
    poly_model=LinearRegression()
    poly_model.fit(X_train_poly,y_train)
    y_poly_pred=poly_model.predict(X_test_poly)

    # Calculate performance
    mse=mean_squared_error(y_test,y_poly_pred)
    r2=r2_score(y_test,y_poly_pred)

    results.append([degree,mse,r2])

    print("\n--- Polynomial Degree",degree,"Results ---")
    print("MSE :",mse)
    print("R2 Score :",r2)

# Display all results
print("\n--- Final Results ---")
print("Linear Regression")
print("MSE :",linear_mse)
print("R2 Score :",linear_r2)

for result in results:
    print("\nPolynomial Degree :",result[0])
    print("MSE :",result[1])
    print("R2 Score :",result[2])

# Plot polynomial regression graphs
for degree in degrees:

    # Create polynomial features for complete dataset
    poly=PolynomialFeatures(degree=degree)
    X_poly=poly.fit_transform(X)

    # Train model
    poly_model=LinearRegression()
    poly_model.fit(X_poly,y)

    # Create smooth X values for graph
    X_plot=pd.DataFrame({
        "displacement":np.linspace(
            X["displacement"].min(),
            X["displacement"].max(),
            300
        )
    })

    X_plot_poly=poly.transform(X_plot)
    y_plot=poly_model.predict(X_plot_poly)

    # Plot actual data and regression curve
    plt.scatter(X,y,color="blue",label="Actual Data")
    plt.plot(
        X_plot,y_plot,
        color="red",
        linewidth=2,
        label="Polynomial Regression"
    )
    plt.xlabel("Engine Displacement")
    plt.ylabel("Miles Per Gallon (MPG)")
    plt.title("Polynomial Regression - Degree "+str(degree))
    plt.legend()
    plt.grid(True)
    plt.show()
