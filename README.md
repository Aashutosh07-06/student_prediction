# 🎓 EduPredict AI — Intelligent Student Performance Analytics & ML Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![MySQL](https://img.shields.io/badge/MySQL-8.0-4479A1.svg)](https://www.mysql.com/)

EduPredict AI is an end-to-end, multi-stakeholder machine learning platform for academic performance forecasting, longitudinal improvement tracking, and institutional decision support. Built with **Streamlit**, **Scikit-Learn**, and **MySQL**.

---

## 🌟 Key Features

### 1. 🔮 Machine Learning Grade Prediction & Scenario Simulation
- Trained on multidimensional academic, behavioral, and psycho-social features (study hours, attendance, assignments, edutech access, stress level, online courses, and discussion forum engagement).
- Instant real-time regression evaluation mapping predictions into Grade Tiers (**Grade A [Honors]**, **Grade B [Good]**, **Grade C [Average]**, **Grade D [At-Risk]**).
- What-If scenario simulations with interactive parameter tuning.

### 2. 📈 Longitudinal Improvement Intelligence ("Says Thus")
- Multi-session performance progression tracking with sequential evaluation delta metrics.
- Dynamic AI Diagnostic Narrative explaining underlying drivers behind score shifts and tier jumps.
- Visual evolution charts:
  - Dual-axis Exam Score & Grade Tier Progression chart.
  - Side-by-side grouped bar comparison (Initial vs. Latest Session).
  - Study hours vs. Exam score trajectory curve.

### 3. 👥 Role-Based Institutional Access Control
- **👨‍🎓 Student Portal**: Individual grade forecasting, personal progression timeline, test history shift logs, and profile management.
- **👩‍🏫 Faculty / Teacher Portal**: Class directory, batch student upload hub, live database analytics, test split model evaluation charts, and individual student improvement inspection.
- **🏛️ Academic Official Portal**: Institution-wide governance dashboard for Deans, Campus Directors, Department Heads (HODs), and Examination Controllers with stakeholder provisioning and cross-department analytics.

### 4. 📁 Batch Class Upload & Bulk Forecasting
- Upload class CSV / Excel rosters for bulk ML inference.
- Instant class grade distributions, summary metrics, and downloadable batch prediction reports.
- Includes a 12-student demo class generator and downloadable CSV templates.

---

## 🗄️ Database Architecture (`edupredict`)

EduPredict AI connects to a relational MySQL backend:
- `users`: Core authentication table with Bcrypt password hashing and role enforcement (`student`, `teacher`, `official`).
- `student_profiles`: Roll numbers, academic course, semester, and section.
- `teacher_profiles`: Employee IDs and departmental affiliations.
- `predictions`: Audit trail of all individual and batch prediction sessions with input vectors, predicted grade scores, and timestamps.

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/Student_Performance_Project.git
cd Student_Performance_Project
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure Database
Create a database in MySQL:
```sql
CREATE DATABASE edupredict;
```
Configure credentials in `database.py` or `.env`.

### 4. Seed Realistic Trajectory Data (Optional)
```bash
python seed_database.py
```

### 5. Launch Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📜 Demo Credentials

| Role | Email | Password | Access Level |
| :--- | :--- | :--- | :--- |
| **Student (Improving)** | `student@gmail.com` | `student123` | Personal Analytics & Trajectory |
| **Teacher** | `teacher@gmail.com` | `teacher123` | Class Audits & Batch Upload |
| **Dean of Academics** | `dean.academics@edupredict.ai` | `dean123` | Institution-wide Governance |
| **Campus Director** | `official@gmail.com` | `official123` | Master Administrative Access |

---

## 📄 License
MIT License. Developed for Academic Performance Analytics & Machine Learning Research.
