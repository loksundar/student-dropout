# 🎓 Student Dropout Prediction System

A web-based machine learning application that predicts student dropout likelihood using demographic, academic, and behavioral features. This system uses **ensemble voting from multiple ML models** to improve prediction accuracy and offers both single-student and batch CSV-based predictions.

✅ **Live Demo** available at: [loksundar.com](https://loksundar.com)

---

## 👤 Target Users

- **Data Scientists** – Analyze student behavior, deploy ensemble ML classifiers.
- **ML/AI Engineers** – Implement voting mechanisms and optimize generalization.
- **Data Engineers** – Prepare labeled datasets and clean educational records.
- **Educational Analysts** – Use the tool to identify at-risk students early.

---

## 🧠 Problem Statement

High dropout rates in schools and colleges pose a major challenge. This system predicts whether a student is at risk of dropping out using structured data (marks, demographics, teachers, internet access, etc.).

---

## ⚙️ Tech Stack

| Component        | Tools Used                                    |
|------------------|-----------------------------------------------|
| Backend          | Python + Flask                                |
| ML Models        | Pickled classifiers: Random Forests, SVMs     |
| Voting Mechanism | Custom majority logic + `scipy.stats.mode`    |
| Batch Inference  | CSV upload support                            |
| Frontend         | HTML + Jinja templates                        |
| Hosting          | [loksundar.com](https://loksundar.com)        |

---

## 📊 Features

- Predict dropout risk for individual students via form input
- Upload CSV files for **bulk predictions**
- Uses **four trained models** (`model1.pkl`, `model3.pkl`, `model4.pkl`, `model5.pkl`)
- Uses **majority voting** to ensure robust predictions
- Outputs list of at-risk students from uploaded datasets

---

## 🧾 Input Features

| Feature              | Description                                |
|----------------------|--------------------------------------------|
| gender               | Male/Female (encoded)                      |
| caste                | Student’s social group (encoded)           |
| mathematics_marks    | Normalized (0-1)                           |
| english_marks        | Normalized (0-1)                           |
| science_marks        | Normalized (0-1)                           |
| science_teacher      | Encoded teacher ID                        |
| languages_teacher    | Encoded teacher ID                        |
| guardian             | Parent/Guardian type (encoded)             |
| internet             | Yes/No for internet access (encoded)       |

---

## 📂 Folder Structure
```bash
├── app.py # Flask app with prediction logic
├── model1.pkl # Trained ML model 1
├── model3.pkl # Trained ML model 3
├── model4.pkl # Trained ML model 4
├── model5.pkl # Trained ML model 5
├── templates/
│ ├── index.html # Form-based prediction page
│ └── data.html # Display results for bulk CSV
```


---

## 🚀 Running Locally

```bash
# Install dependencies
pip install flask pandas numpy scikit-learn scipy

# Run the app
python app.py
---
```
Then go to: http://127.0.0.1:5000

## 🧪 Example Use Case
A school admin inputs scores and info for a student.

Model returns DROP or NOT DROP based on ensemble predictions.

Admin uploads a CSV of 100 students — tool flags IDs of students likely to drop out.

## 🔮 Future Enhancements
Integrate LLMs for free-text reasoning behind dropout predictions

Dashboard view for CSV analytics

Email notifications or interventions for flagged students

Model explainability using SHAP/LIME
