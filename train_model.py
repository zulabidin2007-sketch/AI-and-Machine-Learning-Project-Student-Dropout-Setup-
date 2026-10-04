import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

MODEL_PATH = Path("model.pkl")


def create_training_data():
    data = [
        {"Attendance_Percentage": 95, "Previous_Grades": 90, "Financial_Issues": 0, "Dropout_Risk": 0},
        {"Attendance_Percentage": 92, "Previous_Grades": 85, "Financial_Issues": 0, "Dropout_Risk": 0},
        {"Attendance_Percentage": 88, "Previous_Grades": 82, "Financial_Issues": 1, "Dropout_Risk": 0},
        {"Attendance_Percentage": 86, "Previous_Grades": 78, "Financial_Issues": 1, "Dropout_Risk": 1},
        {"Attendance_Percentage": 80, "Previous_Grades": 72, "Financial_Issues": 1, "Dropout_Risk": 1},
        {"Attendance_Percentage": 75, "Previous_Grades": 70, "Financial_Issues": 1, "Dropout_Risk": 1},
        {"Attendance_Percentage": 68, "Previous_Grades": 60, "Financial_Issues": 1, "Dropout_Risk": 1},
        {"Attendance_Percentage": 60, "Previous_Grades": 55, "Financial_Issues": 1, "Dropout_Risk": 1},
        {"Attendance_Percentage": 90, "Previous_Grades": 88, "Financial_Issues": 0, "Dropout_Risk": 0},
        {"Attendance_Percentage": 82, "Previous_Grades": 76, "Financial_Issues": 0, "Dropout_Risk": 0},
        {"Attendance_Percentage": 70, "Previous_Grades": 62, "Financial_Issues": 1, "Dropout_Risk": 1},
        {"Attendance_Percentage": 58, "Previous_Grades": 50, "Financial_Issues": 1, "Dropout_Risk": 1},
        {"Attendance_Percentage": 55, "Previous_Grades": 48, "Financial_Issues": 1, "Dropout_Risk": 1},
        {"Attendance_Percentage": 94, "Previous_Grades": 91, "Financial_Issues": 0, "Dropout_Risk": 0},
        {"Attendance_Percentage": 87, "Previous_Grades": 83, "Financial_Issues": 0, "Dropout_Risk": 0},
        {"Attendance_Percentage": 63, "Previous_Grades": 58, "Financial_Issues": 1, "Dropout_Risk": 1},
        {"Attendance_Percentage": 52, "Previous_Grades": 45, "Financial_Issues": 1, "Dropout_Risk": 1},
        {"Attendance_Percentage": 98, "Previous_Grades": 96, "Financial_Issues": 0, "Dropout_Risk": 0},
        {"Attendance_Percentage": 84, "Previous_Grades": 80, "Financial_Issues": 0, "Dropout_Risk": 0},
        {"Attendance_Percentage": 66, "Previous_Grades": 57, "Financial_Issues": 1, "Dropout_Risk": 1},
        {"Attendance_Percentage": 72, "Previous_Grades": 65, "Financial_Issues": 1, "Dropout_Risk": 1},
    ]
    return pd.DataFrame(data)


def train_and_save_model():
    df = create_training_data()
    X = df[["Attendance_Percentage", "Previous_Grades", "Financial_Issues"]]
    y = df["Dropout_Risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    with MODEL_PATH.open("wb") as file:
        pickle.dump(model, file)

    print(f"Model saved to {MODEL_PATH}")
    return model


if __name__ == "__main__":
    train_and_save_model()
