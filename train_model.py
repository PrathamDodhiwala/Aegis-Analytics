import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

np.random.seed(42)

size = 3000

fraud_data = pd.DataFrame(
    {
        "Amount": np.random.randint(100, 25000, size),
        "Hour": np.random.randint(0, 24, size),
        "Transaction_Type": np.random.randint(0, 3, size),
        "Location_Risk": np.random.randint(1, 10, size),
        "Device_Risk": np.random.randint(1, 10, size),
        "Fraud": np.random.choice([0, 1], size, p=[0.93, 0.07]),
    }
)

X = fraud_data.drop("Fraud", axis=1)
y = fraud_data["Fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)

model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

joblib.dump(model, "fraud_model.pkl")
joblib.dump(scaler, "scaler.pkl")

fraud_data.to_csv("fraud_dataset.csv", index=False)

print("Training Complete")
