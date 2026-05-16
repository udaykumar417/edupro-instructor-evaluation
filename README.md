# 📚 EduPro – Instructor Performance & Course Quality Evaluation

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-EDA-lightgrey?style=flat-square&logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-orange?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-teal?style=flat-square)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)

> A data-driven framework to evaluate instructor effectiveness, course quality consistency,
> and the relationship between teaching expertise and learner satisfaction on the **EduPro** online education platform.

---

## 📌 Table of Contents
- [Background](#background)
- [Problem Statement](#problem-statement)
- [Dataset Overview](#dataset-overview)
- [Project Structure](#project-structure)
- [Key Analytical Questions](#key-analytical-questions)
- [EDA Highlights](#eda-highlights)
- [Key Insights](#key-insights)
- [Recommendations](#recommendations)
- [How to Run](#how-to-run)
- [Technologies Used](#technologies-used)
- [Author](#author)

---

## 📖 Background

In online education, instructor quality and course design are the strongest drivers of learner satisfaction, course ratings, repeat enrollments, and platform credibility. Even with strong learner demand, poor instructional quality can lead to low course ratings, negative reviews, and reduced trust in the platform.

**EduPro** needed a data-driven framework to evaluate:
- Instructor effectiveness
- Course quality consistency
- The relationship between teaching expertise and learner satisfaction

---

## ❗ Problem Statement

EduPro lacked clarity on:
- Which instructors consistently deliver high-quality courses?
- Does teaching experience translate into better-rated courses?
- Are some course categories more dependent on instructor quality?
- How evenly is teaching performance distributed across the platform?

Without structured analysis, instructor evaluation remained subjective and fragmented.

---

## 🗃️ Dataset Overview

The dataset contains **4 sheets** from the EduPro platform:

| Sheet | Records | Key Fields |
|---|---|---|
| **Teachers** | 60 | TeacherID, Name, Age, Gender, Expertise, YearsOfExperience, TeacherRating |
| **Courses** | 60 | CourseID, Name, Category, Level, Type, Price, Duration, CourseRating |
| **Transactions** | 10,000 | TransactionID, UserID, CourseID, TeacherID, Date, Amount, PaymentMethod |
| **Users** | 3,000 | UserID, Name, Age, Gender, Email |

- **Time Period:** January – December 2025
- **Course Categories:** 12 (Programming, Design, Business, Marketing, Data Science, etc.)
- **Course Levels:** Beginner, Intermediate, Advanced

---

## 📁 Project Structure

```
edupro-instructor-evaluation/
│
├── EduPro_Online_Platform.xlsx   # Raw dataset
├── edupro_eda.py                 # Main EDA script
├── edupro_eda_report.png         # Multi-panel EDA visualization
├── edupro_eda_summary.txt        # Text-based EDA summary & insights
└── README.md                     # Project documentation
```

---

## 🔍 Key Analytical Questions

1. What is the overall distribution of instructor ratings?
2. Do instructors with more experience receive higher ratings?
3. Is there a relationship between TeacherRating and CourseRating?
4. Which expertise areas consistently deliver high-quality courses?
5. Are highly rated instructors associated with higher enrollments?

---

## 📊 EDA Highlights

### KPIs

| KPI | Value |
|---|---|
| Average Teacher Rating | 3.12 / 5.00 |
| Average Course Rating | 3.10 / 5.00 |
| Rating Std Dev (Consistency) | 0.95 |
| Experience–Rating Correlation | 0.598 |
| Teacher→Course Rating Correlation | -0.002 |
| Total Enrollments | 10,000 |

### Instructor Rating Tiers

| Tier | Count |
|---|---|
| Top (4–5) | 11 |
| Good (3–4) | 28 |
| Mid (2–3) | 12 |
| Low (0–2) | 9 |

### Top 3 Instructors

| Name | Expertise | Experience | Rating |
|---|---|---|---|
| Yolanda Levine | Machine Learning | 21 yrs | 4.97 |
| Kimberly Miller | Cybersecurity | 24 yrs | 4.58 |
| Kristi Scott | Machine Learning | 9 yrs | 4.39 |

### Course Rating by Category (Top & Bottom)

| Category | Avg Rating |
|---|---|
| Marketing | 3.72 ✅ |
| Digital Marketing | 3.66 ✅ |
| Machine Learning | 2.66 ❌ |
| Business | 2.69 ❌ |

---

## 💡 Key Insights

1. **Experience matters (r = 0.598):** More experienced instructors consistently earn higher ratings — the strongest driver of teacher quality in the data.

2. **Teacher rating ≠ Course rating (r ≈ 0):** Virtually zero correlation between instructor rating and course rating — course quality is driven by curriculum design, not just instructor personality.

3. **Top-tier teachers dominate enrollments:** 68% of all 10,000 enrollments go to Top-tier (4–5 rated) instructors, even though they are only 18% of the teacher pool.

4. **Marketing & Machine Learning** have the strongest instructor quality; **Business & Design** lag significantly behind.

5. **Advanced courses are rated lowest** (avg 2.81 vs 3.32 for Intermediate) — suggesting curriculum review is needed at the advanced level.

6. **9 instructors are in the Low tier (< 2.0)** — a clear and immediate target for performance intervention.

7. **Female instructors slightly outperform male instructors** on average (3.23 vs 3.04).

---

## ✅ Recommendations

- **Fast-track mentoring** for the 9 instructors rated below 2.0.
- **Investigate curriculum gaps** in Business and Design categories.
- **Review Advanced-level course content** across all categories.
- **Leverage top instructors** (Yolanda Levine, Kimberly Miller) as mentors or course design consultants.
- **Develop structured onboarding** for new instructors — early-career teachers cluster in lower rating tiers.

---

## ▶️ How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn openpyxl
```

### Run EDA
```bash
python edupro_eda.py
```

**Outputs:**
- `edupro_eda_report.png` — Multi-panel visualization (10 charts)
- `edupro_eda_summary.txt` — Full text summary with insights

---

## 🤖 Machine Learning Models

This project includes 5 ML models for advanced analysis:

| Model | Algorithm | R² Score | Purpose |
|---|---|---|---|
| **Teacher Rating Prediction** | Random Forest Regressor | 0.517 | Predict instructor quality from demographics |
| **Course Rating Prediction** | Random Forest Regressor | -0.119 | Evaluate course quality drivers |
| **Enrollment Forecasting** | Gradient Boosting | -0.887 | Forecast enrollment based on instructor features |
| **Instructor Clustering** | K-Means (k=4) | - | Identify 4 instructor performance profiles |
| **Feature Importance** | Random Forest | - | Rank factors driving instructor & enrollment |

**Key ML Findings:**
- YearsOfExperience is the #1 predictor of TeacherRating (58% importance)
- CourseCategory is the #1 predictor of CourseRating (64% importance)
- 4 distinct instructor clusters: Elite Performers, Solid Performers, Rising Talent, Needs Support
- TeacherRating is the top driver of enrollment (after experience)

**Run ML Models:**
```bash
python edupro_ml.py
```
Outputs:
- `edupro_ml_report.png` — 10-panel ML visualizations
- `edupro_ml_summary.txt` — Model metrics & insights

## 🛠️ Technologies Used

- **Python 3.10+**
- **Pandas** — Data manipulation & merging
- **NumPy** — Numerical computations
- **Scikit-Learn** — Machine Learning models (Random Forest, K-Means, Gradient Boosting)
- **Matplotlib** — Visualization
- **Seaborn** — Statistical plots
- **OpenPyXL** — Excel file reading

---

## 👤 Author

**[UDAY KUMAR]**
Virtual Intern — Machine Learning
Unified Mentor Internship Program

---

> *This project was completed as part of the Unified Mentor Virtual Internship Program.*
