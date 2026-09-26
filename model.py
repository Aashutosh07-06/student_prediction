import joblib
import pandas as pd


MODEL_PATH = "model/student_model.pkl"


model = joblib.load(
    MODEL_PATH
)


FEATURES = [
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


def predict_grade(data):

    df = pd.DataFrame(
        [data],
        columns=FEATURES
    )

    prediction = model.predict(df)[0]

    # Keep prediction between 0 and 3
    prediction = max(
        0,
        min(3, prediction)
    )

    return float(prediction)


def grade_label(prediction):

    prediction = round(
        prediction
    )

    grade_mapping = {
        0: "A",
        1: "B",
        2: "C",
        3: "D"
    }

    return grade_mapping[prediction]