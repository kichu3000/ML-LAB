# Program 4: MLE and MAP Logistic Regression
import warnings
warnings.filterwarnings("ignore")
# Import required libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error

# Load the Breast Cancer Wisconsin dataset
data = load_breast_cancer()
X = data.data
y = data.target

print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MLE Logistic Regression
# penalty=None means no regularization
mle_model = LogisticRegression(
    penalty=None,
    max_iter=10000
)

mle_model.fit(X_train, y_train)

# Make predictions
mle_pred = mle_model.predict(X_test)

# Calculate accuracy
mle_accuracy = accuracy_score(y_test, mle_pred)

# MAP with L2 Prior
l2_model = LogisticRegression(
    penalty="l2",
    C=1,
    max_iter=10000
)

l2_model.fit(X_train, y_train)

# Make predictions
l2_pred = l2_model.predict(X_test)

# Calculate accuracy
l2_accuracy = accuracy_score(y_test, l2_pred)

# MAP with L1 Prior
l1_model = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    C=1,
    max_iter=10000
)

l1_model.fit(X_train, y_train)

# Make predictions
l1_pred = l1_model.predict(X_test)

# Calculate accuracy
l1_accuracy = accuracy_score(y_test, l1_pred)

# Compare model performance
print("\nMODEL PERFORMANCE")
print("MLE Accuracy:", mle_accuracy)
print("MAP-L2 Accuracy:", l2_accuracy)
print("MAP-L1 Accuracy:", l1_accuracy)

# Compare parameter estimates
print("\nPARAMETER ESTIMATES")

print("MLE coefficients:")
print(mle_model.coef_)

print("\nMAP-L2 coefficients:")
print(l2_model.coef_)

print("\nMAP-L1 coefficients:")
print(l1_model.coef_)