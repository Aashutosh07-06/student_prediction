import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = "data/merged_dataset.csv"


def load_data():

    return pd.read_csv(
        DATA_PATH
    )


def dataset_summary():

    df = load_data()

    return {
        "students": len(df),
        "features": len(df.columns),
        "average_exam": round(
            df["ExamScore"].mean(),
            2
        ),
        "average_attendance": round(
            df["Attendance"].mean(),
            2
        ),
        "average_study_hours": round(
            df["StudyHours"].mean(),
            2
        )
    }


def study_hours_chart():

    df = load_data()

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.scatter(
        df["StudyHours"],
        df["ExamScore"],
        alpha=0.4
    )

    ax.set_xlabel(
        "Study Hours"
    )

    ax.set_ylabel(
        "Exam Score"
    )

    ax.set_title(
        "Study Hours vs Exam Score"
    )

    return fig


def attendance_chart():

    df = load_data()

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.scatter(
        df["Attendance"],
        df["ExamScore"],
        alpha=0.4
    )

    ax.set_xlabel(
        "Attendance (%)"
    )

    ax.set_ylabel(
        "Exam Score"
    )

    ax.set_title(
        "Attendance vs Exam Score"
    )

    return fig


def grade_distribution():

    df = load_data()

    counts = df["FinalGrade"].value_counts().sort_index()

    labels = [
        "A",
        "B",
        "C",
        "D"
    ]

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.bar(
        labels,
        [
            counts.get(i, 0)
            for i in range(4)
        ]
    )

    ax.set_xlabel(
        "Final Grade"
    )

    ax.set_ylabel(
        "Number of Students"
    )

    ax.set_title(
        "Final Grade Distribution"
    )

    return fig