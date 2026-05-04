import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def train_heart_disease_model():
    # Load dataset
    heart = pd.read_csv(r"C:\Users\nisha\Downloads\heart_disease_data.csv")

    # Split input & output
    X = heart.drop("target", axis=1)
    y = heart["target"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train model
    heart_model = LogisticRegression(max_iter=1000)
    heart_model.fit(X_train, y_train)

    # Save model
    joblib.dump(heart_model, "heart_model.pkl")
    print("Heart disease model trained and saved.")

def train_diabetes_model():
    # Load dataset
    diabetes = pd.read_csv(r"C:\Users\nisha\Downloads\diabetes.csv")

    # Split input & output
    X = diabetes.drop("Outcome", axis=1)
    y = diabetes["Outcome"]

    # Data Standardization
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

    # Create and train model (SVM)
    diabetes_model = svm.SVC(kernel='linear')
    diabetes_model.fit(X_train, y_train)

    # Save model and scaler
    joblib.dump(diabetes_model, "diabetes_model.pkl")
    joblib.dump(scaler, "diabetes_scaler.pkl")
    print("Diabetes model trained and saved.")

if __name__ == "__main__":
    train_heart_disease_model()
    train_diabetes_model()
    print("All models trained successfully!")