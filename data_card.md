### Data Card - HR Dataset Overview

**Dataset Description**
This dataset, titled HRDataset_v14.csv, contains structured information for 311 individual employees. It is designed to support the development of models that predict employee attrition. The core task is a binary classification to determine if an employee is likely to stay or leave the company.

**Key Features**
* **Financial and Professional:** The data tracks annual salary and specific job positions such as Production Technician, Software Engineer, or Data Analyst
* **Engagement and Performance:** It includes internal metrics like engagement survey scores, official performance scores (ranging from "Exceeds" to "PIP"), and employee satisfaction levels
* **Operational:** Records show the department an employee belongs to, their direct manager, and how many absences they have accumulated.
* **Temporal:** Critical dates are included, such as the date of hire, date of termination (if applicable), and the date of the last performance review.
* **Recruitment:** The source of hire is tracked, including platforms like LinkedIn, Indeed, and employee referrals.

**Target Variable**
The primary target is the "Termd" column. A value of 1 signifies that the employee has left the organization, while a value of 0 indicates they are currently active in their role.

**Bias and Reliability Limitations**
* **Demographic Representation:** The data includes protected characteristics such as sex, race, and marital status. While these are often removed during model training to promote fairness, users should be aware that indirect biases can still exist in other variables like salary or position.
* **Sample Size for Minorities:** Certain groups, such as "American Indian or Alaska Native" or "Two or more races," represent a very small portion of the total 311 records. This lack of data can lead to less reliable predictions for these specific individuals.
* **Subjectivity:** Performance scores and reasons for leaving (e.g., "unhappy" or "career change") are based on human judgment and may reflect the subjective views of past managers.

**Recommended Usage**
* **Strategic Retention:** This data should be used to help HR teams identify broad trends and potential turnover risks before they happen.
* **Support, Not Replacement:** The dataset is intended for decision support. It should not be used to automate high-stakes HR actions like firing or hiring without significant human review and validation.
* **Organization Specific:** Because this data reflects the culture and structure of one specific company, it may not accurately predict behavior in different industries or geographic regions.