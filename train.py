import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("Mobile.csv")

print("Dataset Loaded Successfully ✅")
print("Shape:", df.shape)

# -----------------------------
# Clean Required Columns
# -----------------------------
columns_to_clean = [
    "Ram_mb",
    "Internal_memeory_gb",
    "Primary_camera",
    "Battery_power_mAh"
]

for col in columns_to_clean:
    df[col] = df[col].str.replace(r'[^0-9.]', '', regex=True)
    df[col] = pd.to_numeric(df[col])

df["Speed_of_microprocessor"] = pd.to_numeric(df["Speed_of_microprocessor"])

# -----------------------------
# Convert price_range to numeric
# -----------------------------
price_mapping = {
    "Low cost": 0,
    "Medium cost": 1,
    "High cost": 2,
    "Very High cost": 3
}

df["price_range"] = df["price_range"].map(price_mapping)

# -----------------------------
# Select Features
# -----------------------------
features = [
    "Ram_mb",
    "Internal_memeory_gb",
    "Speed_of_microprocessor",
    "Primary_camera",
    "Battery_power_mAh"
]

X = df[features]
y = df["price_range"]

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Initialize Models
# -----------------------------
log_model = LogisticRegression(max_iter=2000)
knn_model = KNeighborsClassifier()
dt_model = DecisionTreeClassifier(random_state=42)

# -----------------------------
# Train Models
# -----------------------------
log_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)

# -----------------------------
# Calculate Accuracy
# -----------------------------
log_acc = accuracy_score(y_test, log_model.predict(X_test))
knn_acc = accuracy_score(y_test, knn_model.predict(X_test))
dt_acc = accuracy_score(y_test, dt_model.predict(X_test))

print("\nModel Accuracies 📊")
print("Logistic Regression:", log_acc)
print("KNN:", knn_acc)
print("Decision Tree:", dt_acc)

# -----------------------------
# Select Best Model
# -----------------------------
accuracies = {
    "Logistic Regression": log_acc,
    "KNN": knn_acc,
    "Decision Tree": dt_acc
}

best_model_name = max(accuracies, key=accuracies.get)
print("\nBest Model:", best_model_name)

if best_model_name == "Logistic Regression":
    best_model = log_model
elif best_model_name == "KNN":
    best_model = knn_model
else:
    best_model = dt_model

# Save Best Model
pickle.dump(best_model, open("model.pkl", "wb"))

print("\nBest Model Saved Successfully ✅")