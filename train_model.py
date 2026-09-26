import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================
# LOAD DATASET
# ============================================

df = pd.read_csv(
    "data/merged_dataset.csv"
)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================
# FEATURES
# ============================================

features = [
    "StudyHours",
    "Attendance",
    "Resources",
    "Extracurricular",
    "Motivation",
    "Internet",
    "Gender",
    "Age",
    "LearningStyle",
    "OnlineCourses",
    "Discussions",
    "AssignmentCompletion",
    "ExamScore",
    "EduTech",
    "StressLevel"
]


target = "FinalGrade"


X = df[features]

y = df[target]


# ============================================
# TRAIN / TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================
# CREATE LINEAR REGRESSION MODEL
# ============================================

model = LinearRegression()


# ============================================
# TRAIN
# ============================================

model.fit(
    X_train,
    y_train
)


print("\nModel training completed!")


# ============================================
# PREDICTION
# ============================================

y_pred = model.predict(
    X_test
)


# ============================================
# EVALUATION
# ============================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n================================")
print("MODEL EVALUATION")
print("================================")

print(
    "MAE:",
    round(mae, 4)
)

print(
    "MSE:",
    round(mse, 4)
)

print(
    "R2 Score:",
    round(r2, 4)
)


# ============================================
# SAVE MODEL
# ============================================

joblib.dump(
    model,
    "model/student_model.pkl"
)


print("\n================================")
print("MODEL SAVED")
print("================================")

print(
    "model/student_model.pkl"
)