# Explainable & Ethical AI for Employee Attrition Prediction

## Objectives
The goal of this project is to create a reliable artificial intelligence system capable of predicting employee departures while guaranteeing:
- Explainability: understanding why the model makes decisions

- Fairness: avoiding bias across demographic groups

- Transparency: making the model interpretable for HR stakeholders

- Ethics: ensuring responsible use of predictions

## Scope
This project focuses on:

- Predicting whether an employee is likely to leave the company (Termd)

- Analyzing HR-related factors such as:

  - Job satisfaction

  - Engagement level

  - Absenteeism

  - Salary

  - Tenure

The system is designed for decision support, not automated decision-making.

## Persona (Target Users)
Primary users:

- HR Managers

- Talent Retention Teams

- Organizational Analysts

Use case:

- Identify employees at risk of leaving

- Improve retention strategies

- Support HR decision-making with explainable insights

## Project Pipeline
Data Collection
      ↓
Data Cleaning & Validation
      ↓
Feature Engineering (Tenure, Review Recency)
      ↓
Encoding & Scaling
      ↓
Model Training (Classification)
      ↓
Explainability (Feature Importance / SHAP)
      ↓
Evaluation & Bias Analysis

## Instructions to Run the Project
1. Clone the repository
   git clone https://github.com/harreshh05/Explainability-AI.git
   cd Explainability-AI

2. Install dependencies
   pip install -r requirements.txt

3. Run the Notebook and Import the Data sets
   Open:
     proto_v2.ipynb

4. Execute pipeline
- Run preprocessing cells

- Run model training

- View evaluation and explainability results

## Outputs
- Employee attrition prediction (0 = stay, 1 = leave)

- Feature importance analysis

- Explainability insights

## ⚠️ Ethical Considerations

- Personal identifiers removed (e.g., names, IDs)

- Bias analysis performed (gender, etc.)

- Model used for decision support only, not automation

## 📁 Project Structure
├── HRDataset_v14.csv

├── README.md

├── proto_v2.ipynb

├── requirements.txt

├── technical_documentation.md

├── model_card.md

├── data_card.md


