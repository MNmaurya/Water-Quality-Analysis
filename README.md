# 💧 Explainable AI-Based Water Quality Assessment System

An interactive Machine Learning application that predicts water-quality categories using historical monitoring data and explains its predictions using Explainable Artificial Intelligence (XAI).

The system combines machine learning classification, SHAP-based explanations, and indicative water-use recommendations to make water-quality assessment more understandable and accessible.

---

## 📌 Project Overview

Water quality is an important factor affecting environmental sustainability and public health. Traditional water-quality assessment requires the analysis of multiple physical, chemical, and biological parameters.

This project uses Machine Learning to analyse water-quality parameters and classify water samples into four categories:

- Excellent
- Good
- Poor
- Very Poor

In addition to predicting the water-quality category, the application provides SHAP-based explanations to identify the features contributing to the prediction and displays indicative recommendations regarding potential water uses.

The project is developed as a software-based application using Python and Streamlit.

---

## 🎯 Objectives

- Develop a Machine Learning-based water-quality classification system.
- Analyse multiple water-quality parameters.
- Compare different classification algorithms.
- Select the best-performing model using evaluation metrics.
- Implement Explainable AI using SHAP.
- Provide understandable explanations of individual predictions.
- Display indicative water-use recommendations based on predicted categories.
- Build an interactive dashboard for users.

---

## ✨ Key Features

### 1. Water Quality Prediction
Predicts one of four water-quality categories:
- Excellent
- Good
- Poor
- Very Poor

### 2. Interactive Dashboard
Provides an easy-to-use interface with numerical sliders and categorical dropdown menus for entering water-quality parameters.

### 3. Model Class Probabilities
Displays the model's estimated probabilities for each predicted class.

### 4. Explainable AI Using SHAP
Uses SHAP (SHapley Additive exPlanations) to explain the contribution of individual input features to the model's prediction.

A horizontal bar chart displays the most influential features for an individual prediction.

### 5. Recommended Water Uses
Displays indicative potential uses and precautions according to the predicted water-quality category.

### 6. Dataset Overview
Displays:
- Total monitoring stations
- Number of input features
- Number of water-quality classes
- Water-quality class distribution

---

## 📊 Dataset Description

The project uses a processed water-quality monitoring dataset containing:

| Dataset Information | Details |
|---|---|
| Total Records | 683 |
| Total Columns | 17 |
| Input Features | 15 |
| Target Variable | Water_Quality_Class |
| Number of Classes | 4 |

### Water Quality Parameters

The dataset includes the following parameters:

| Parameter | Description |
|---|---|
| Temperature | Average water temperature |
| DO | Dissolved Oxygen |
| pH | Acidity or alkalinity of water |
| Conductivity | Electrical conductivity of water |
| BOD | Biochemical Oxygen Demand |
| Nitrate | Nitrate concentration |
| FColi | Faecal Coliform |
| TColi | Total Coliform |
| Mean Features | Average observed parameter values |
| Range Features | Variation in parameter values |
| State | Geographical information |
| Water_Body_Type | Type of monitored water body |

The dataset also contains a target column, `Water_Quality_Class`, representing the four water-quality categories.

### Target Class Distribution

| Water Quality Class | Records |
|---|---:|
| Excellent | 156 |
| Good | 213 |
| Poor | 142 |
| Very Poor | 172 |
| **Total** | **683** |

`Station_ID` is excluded from model training because it serves as an identifier.

---

## ⚙️ Machine Learning Methodology

Three classification algorithms were implemented and evaluated.

### Algorithms Used

**1. Logistic Regression**
- Used as a baseline classification algorithm.
- Estimates probabilities for different classes.

**2. Random Forest Classifier**
- Uses an ensemble of decision trees.
- Combines multiple tree predictions.
- Captures complex relationships between input features.

**3. Extra Trees Classifier**
- Uses an ensemble of highly randomized decision trees.
- Introduces additional randomness during tree construction.

### Model Evaluation

The models were evaluated using:
- Accuracy
- Balanced Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Confusion Matrix
- Classification Report

### Model Comparison

| Algorithm | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| Random Forest | 96.35% | 96.25% | 96.39% |
| Extra Trees | 89.78% | 89.89% | 90.04% |
| Logistic Regression | 74.45% | 75.94% | 75.22% |

**Selected Model: Random Forest Classifier**

Random Forest achieved the highest Macro F1-score and was selected as the final model.

The dataset was divided into 80% training data and 20% testing data using a stratified split with `random_state=42`.

---

## 🧠 Explainable AI (SHAP)

A key component of this project is Explainable Artificial Intelligence.

SHAP stands for **SHapley Additive exPlanations**. It is based on Shapley values from cooperative game theory.

Instead of displaying only a predicted category, SHAP helps explain how individual input features contributed to the model's prediction.

The application uses:
- SHAP TreeExplainer
- The trained tree-based model
- Transformed input features
- Feature contribution values
- Horizontal bar-chart visualisation

This makes the prediction process more interpretable and helps users understand the model's behaviour.

---

## 💧 Water Use Recommendations

The application includes a rule-based recommendation component that displays indicative potential uses and precautions based on the predicted water-quality category.

| Predicted Class | Indicative Recommendation |
|---|---|
| Excellent | General uses such as gardening, washing and cleaning, subject to applicable safety requirements |
| Good | Selected non-potable uses, subject to suitability and testing |
| Poor | Limited potential uses; further assessment and treatment required |
| Very Poor | Not recommended for direct use; further assessment and appropriate treatment required |

These recommendations are indicative and are not a substitute for laboratory testing or regulatory water-quality certification.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Jupyter Notebook | Data analysis and model development |
| Pandas | Data manipulation |
| NumPy | Numerical computations |
| Matplotlib | Data visualisation |
| Seaborn | Exploratory visualisation |
| Scikit-learn | Preprocessing, model training and evaluation |
| SHAP | Explainable AI |
| Streamlit | Interactive web application |
| Joblib | Model saving and loading |
| JSON | Model metadata storage |

---

## 📂 Project Structure

```text
water_quality_project/
│
├── app.py
├── water_quality_dataset.csv
├── water_quality_model.joblib
├── metadata.json
├── model_comparison.csv
├── classification_report.csv
├── requirements.txt
│
└── README.md
```

### File Description

- `app.py` – Main Streamlit application.
- `water_quality_dataset.csv` – Processed water-quality dataset.
- `water_quality_model.joblib` – Saved trained Machine Learning pipeline.
- `metadata.json` – Stores model and feature-related metadata.
- `model_comparison.csv` – Model evaluation comparison.
- `classification_report.csv` – Detailed classification results.
- `requirements.txt` – Required Python libraries and dependencies.
- `README.md` – Project documentation.

---
The application will open in your default browser. If it does not, open the local URL displayed in the terminal.

**Important:** Keep all required project files in the same directory so that the application can load the saved model, metadata, and dataset.

---

## 📈 Project Workflow

```text
Water Quality Dataset
         |
         v
Data Exploration and Analysis
         |
         v
Data Preprocessing
         |
         v
Train-Test Split
         |
         v
Model Training
         |
         v
Model Evaluation and Selection
         |
         v
Saved Random Forest Pipeline
         |
         v
Streamlit Dashboard
         |
         v
User Enters Water Parameters
         |
         v
Water Quality Prediction
         |
         +----------------------+
         |                      |
         v                      v
    SHAP Explanation      Water Use Recommendations
         |                      |
         +----------+-----------+
                    |
                    v
             Final Dashboard
```

---

## ⚠️ Limitations and Disclaimer

- The model is trained on a dataset containing 683 monitoring-station records.
- Prediction quality depends on the available dataset and its underlying class definitions.
- Model performance on unseen geographical regions or different water sources may vary.
- The system predicts dataset-defined water-quality categories.
- SHAP explains model behaviour but does not establish causation.
- The application does not perform real-time water testing.
- The water-use recommendations are indicative and should not be treated as regulatory certification.
- A model prediction must not be used as proof that water is safe for drinking. Appropriate laboratory testing and applicable standards are necessary to establish drinking-water suitability.

---

## 🔮 Future Scope

Possible future enhancements include:

- Integration with IoT-based water-quality sensors.
- Real-time water-quality monitoring.
- Larger and more diverse datasets.
- Historical water-quality trend analysis.
- Integration with updated regulatory standards.
- Advanced parameter-wise analysis.
- Improved validation using independent datasets.
- Deployment as a publicly accessible web application.

---

## 🎓 Academic Project

**Project Title:** Explainable AI-Based Water Quality Assessment System

**Domain:** Artificial Intelligence and Machine Learning

**Project Type:** Academic Machine Learning Project

**Developed using:** Python, Scikit-learn, SHAP and Streamlit

---

## 👩‍💻 Author

**Muskan Maurya**

B.Tech – Artificial Intelligence

Bhai Parmanand Delhi Skill and Entrepreneurship University (BPDSEU)

---

*Developed as an academic project to explore the application of Machine Learning and Explainable AI in water-quality assessment.*
