### Model Card: Employee Attrition Prediction System

**1. Model Objective**
* This model aims to predict employee attrition, defined as whether an employee is likely to leave the organization.
* It is designed as a decision-support tool for HR teams, enabling proactive retention strategies and improved workforce management.
* Inputs consist of structured HR data such as salary, tenure, performance metrics, and engagement indicators.
* The output is a binary prediction (stay or leave) along with a probability score indicating risk level.

**2. Training Data**
* The model is trained on the HRDataset_v14.csv dataset, containing employee records across various roles and departments.
* The dataset includes engineered features such as tenure and time since last review.
* Sensitive attributes (e.g., name, gender, race) were removed during training to reduce bias risks.
* Data preprocessing includes cleaning, encoding categorical variables, and scaling numerical features.
* A stratified train-test split (80/20) was applied to preserve class distribution.

**3. Performance**
* Several models were evaluated, including Logistic Regression, SVM, Decision Trees, Random Forest, Gradient Boosting, and XGBoost.
* The best-performing model was a tuned Random Forest classifier.
* Evaluation metrics include Accuracy, Precision, Recall, F1-score, and ROC-AUC.
* The model achieves strong performance, with ROC-AUC estimated between 0.85 and 0.90.
* Recall is prioritized to ensure high detection of at-risk employees.

**4. Limitations**
* The model may struggle with employees who have limited historical data, such as new hires.
* It does not capture external factors such as economic conditions or personal motivations.
* There is potential for indirect bias through proxy variables such as salary or tenure.
* Performance may degrade on out-of-distribution data.

**5. Risks and Mitigation**
* The model must not be used for automated HR decisions such as termination.
* It is intended strictly for decision support, with human oversight required.
* Explainability techniques (e.g., feature importance, SHAP) are used to interpret predictions.
* Bias analysis is conducted to identify and mitigate unfair outcomes.

**6. Energy and Efficiency**
* The model uses a Random Forest algorithm, which has moderate training cost due to ensemble learning and hyperparameter tuning.
* Inference is efficient and can be performed in milliseconds on standard CPU hardware.
* Overall energy consumption is relatively low compared to deep learning approaches.

**7. Security**
* All inputs are validated and cleaned to prevent invalid or malicious data.
* Personally identifiable information has been removed to ensure privacy.
* Sensitive information such as API keys should be stored securely using environment variables.