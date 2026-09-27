import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
df = pd.read_csv("data/diabetes.csv")

# Remove missing values
df.dropna(inplace=True)

# Separate features and target
x = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Logistic Regression without feature scaling
model = LogisticRegression(max_iter=10000)
model.fit(x_train, y_train)

pred = model.predict(x_test)

accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)

print("\n---- Logistic Regression Without Feature Scaling ----")
print("\n---- MODEL PERFORMANCE ----")
print("Accuracy  : ", accuracy)
print("Precision : ", precision)
print("Recall    : ", recall)
print("F1 Score  : ", f1)

# Confusion matrix
cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Diabetes", "Diabetes"]
)
disp.plot()
plt.title("Without Feature Scaling")
plt.show()

# Apply feature scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train the same model with scaled data
model.fit(x_train_scaled, y_train)

pred_scaled = model.predict(x_test_scaled)

accuracy_scaled = accuracy_score(y_test, pred_scaled)
precision_scaled = precision_score(y_test, pred_scaled)
recall_scaled = recall_score(y_test, pred_scaled)
f1_scaled = f1_score(y_test, pred_scaled)

print("\n---- Logistic Regression With Feature Scaling ----")
print("\n---- MODEL PERFORMANCE ----")
print("Accuracy  : ", accuracy_scaled)
print("Precision : ", precision_scaled)
print("Recall    : ", recall_scaled)
print("F1 Score  : ", f1_scaled)

# Confusion matrix
cm_scaled = confusion_matrix(y_test, pred_scaled)
disp_scaled = ConfusionMatrixDisplay(
    confusion_matrix=cm_scaled,
    display_labels=["No Diabetes", "Diabetes"]
)
disp_scaled.plot()
plt.title("With Feature Scaling")
plt.show()

