# Program 3: Comparison of Linear, Ridge and Lasso Regression

# Import required libraries
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load the Diabetes dataset
data = load_diabetes()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standard Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Make predictions
linear_prediction = linear_model.predict(X_test)

# Calculate performance metrics
linear_mse = mean_squared_error(y_test, linear_prediction)
linear_r2 = r2_score(y_test, linear_prediction)

# Ridge Regression with cross-validation
ridge_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge())
])

# Define alpha values for tuning
ridge_params = {
    "ridge__alpha": [0.01, 0.1, 1, 10, 100]
}

# Perform 5-fold cross-validation
ridge_grid = GridSearchCV(
    ridge_pipeline,
    ridge_params,
    cv=5,
    scoring="neg_mean_squared_error"
)

# Train Ridge model
ridge_grid.fit(X_train, y_train)

# Make predictions
ridge_prediction = ridge_grid.predict(X_test)

# Calculate performance metrics
ridge_mse = mean_squared_error(y_test, ridge_prediction)
ridge_r2 = r2_score(y_test, ridge_prediction)

# Lasso Regression with cross-validation
lasso_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", Lasso(max_iter=10000))
])

# Define alpha values for tuning
lasso_params = {
    "lasso__alpha": [0.01, 0.1, 1, 10, 100]
}

# Perform 5-fold cross-validation
lasso_grid = GridSearchCV(
    lasso_pipeline,
    lasso_params,
    cv=5,
    scoring="neg_mean_squared_error"
)

# Train Lasso model
lasso_grid.fit(X_train, y_train)

# Make predictions
lasso_prediction = lasso_grid.predict(X_test)

# Calculate performance metrics
lasso_mse = mean_squared_error(y_test, lasso_prediction)
lasso_r2 = r2_score(y_test, lasso_prediction)

# Display results
print("STANDARD LINEAR REGRESSION")
print("MSE:", linear_mse)
print("R2:", linear_r2)

print("\nRIDGE REGRESSION")
print("Best Alpha:", ridge_grid.best_params_["ridge__alpha"])
print("MSE:", ridge_mse)
print("R2:", ridge_r2)

print("\nLASSO REGRESSION")
print("Best Alpha:", lasso_grid.best_params_["lasso__alpha"])
print("MSE:", lasso_mse)
print("R2:", lasso_r2)