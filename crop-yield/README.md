# AgroTech Innovations Machine Learning Pipeline

## Full Name and Email Address
Full Name: Noel Rodrigues
Email Address: noel.mr@gmail.com

## Project Overview
This project aims to develop machine learning models to address key challenges faced by AgroTech Innovations, a leading agri-tech company. The goal is to help optimize crop yields and resource management in controlled farming environments. Specifically, the pipeline is designed to predict temperature conditions and categorize plant types and stages using sensor data.

The project includes an end-to-end machine learning pipeline, from data preprocessing to model training, evaluation, and potential deployment.

## Folder Structure Overview
/AgroTech-ML-Pipeline
│
├── data/                    # Folder containing the agri.db dataset
│   └── agri.db              # Database containing sensor data
│
├── eda.ipynb               # Jupyter notebook for exploratory data analysis (EDA)
│
├── src/                     # Source code for the machine learning pipeline
│   └── end_to_end_ml_pipeline.py           # Main pipeline orchestration script
│
├── config/                  # Configuration files for easy experiment management
│   └── config.yaml          # Configuration file for setting pipeline parameters
│
├── requirements.txt         # List of dependencies
└── README.md                # This file

## Installation Instructions
1. Clone the Repository: To get started with this project, clone the repository to your local machine:
git clone <https://github.com/rodriguesnoel/aiap19-noel-rodrigues-991H>

2. Install Dependencies: Create a virtual environment (optional but recommended) and install the required packages using the requirements.txt:
pip install -r requirements.txt

3. Dataset: The dataset agri.db can be accessed at the following URL: <https://techassessment.blob.core.windows.net/aiap19-assessment-data/agri.db>
Ensure that the agri.db file is placed in the data/ directory of your project folder.

## Pipeline Execution and Modifying Parameters
The pipeline consists of the following logical steps: data preprocessing, model training, hyperparameter tuning, evaluation, and potential deployment. You can configure various parameters for each stage by modifying the configuration file (config/config.yaml).

To run the pipeline, simply execute the following script from the command line:
python src/end_to_end_ml_pipeline.py

You can also modify the parameters for model training, evaluation, or preprocessing by adjusting the configuration file. More details on configurable parameters can be found in config/config.yaml

## Pipeline Logical Flow
The pipeline follows these steps:
1. Data Import: Data is fetched from the agri.db SQLite database.
2. Data Preprocessing:
    - Cleaning the dataset (handling missing values, outliers, etc.)
    - Feature engineering (e.g., combining temperature and humidity data)
3. Model Training:
    - Training multiple models (e.g., regression models for temperature prediction, classification models for plant categorization)
    - Hyperparameter tuning to optimize model performance
4. Model Evaluation:
    - Evaluate model performance using appropriate metrics (e.g., RMSE, accuracy, precision, recall)
5. Model Deployment: Optional step for deploying models as APIs or saving the models for future use.
    - A flowchart visualizing this process is provided in the eda.ipynb file.

## Exploratory Data Analysis (EDA) Summary
In this project, exploratory data analysis (EDA) was conducted to understand the distribution of sensor readings and their relationship with plant conditions. Key findings include:

- Temperature and Humidity: Strong correlation found between temperature and humidity, which will be useful for predicting optimal environmental conditions.
- Feature Engineering: Combined multiple sensor readings (e.g., light intensity and CO2 levels) to create derived features that improve model performance.

Detailed steps and visualizations of the EDA can be found in the Jupyter notebook (eda.ipynb).

## Feature Processing
The dataset contains several raw sensor features, including temperature, humidity, light intensity, CO2 levels, and nutrient concentrations. These features are processed as follows:

| Feature Name     | Description                                   | Transformation or Processing                 |
|------------------|-----------------------------------------------|----------------------------------------------|
| Temperature      | Sensor reading of temperature in the farm     | Normalization, outlier handling              |
| Humidity         | Sensor reading of humidity levels             | Normalization, outlier handling              |
| Light Intensity  | Sensor reading of light intensity             | Scaling, missing value imputation            |
| CO2 Levels       | Sensor reading of CO2 concentration           | Scaling, outlier handling                    |
| Nutrient Levels  | Sensor reading of nutrient concentrations     | Log transformation (if skewed)               |


## Model Selection and Rationale
For each task, the following models were evaluated:

1. Temperature Prediction:
- Linear Regression: Used as a baseline model due to its simplicity and interpretability.
- Random Forest Regressor: Chosen for its ability to capture non-linear relationships and handle complex feature interactions.

2. Plant Type-Stage Classification:
- Logistic Regression: Chosen for its simplicity and ease of interpretation for multi-class classification.
- Random Forest Classifier: Selected for its robustness in classification tasks with complex feature interactions.

These models were selected based on the nature of the problem (regression vs classification) and the complexity of the data. Hyperparameter tuning was performed using cross-validation.

## Model Evaluation
Model performance was evaluated using the following metrics:
- Regression Task: Root Mean Squared Error (RMSE) was used to evaluate temperature prediction accuracy.
- Classification Task: Accuracy, Precision, Recall, and F1-Score were used to evaluate the classification of plant types and stages.
For each model, cross-validation was employed to ensure robust performance estimates.

## Considerations for Model Deployment
While this project is a submission assessment, the pipeline is designed with potential deployment in mind. The following considerations were made for future deployment:

1. Scalability: The pipeline can be adapted for use in cloud-based systems by containerizing the models (e.g., using Docker).
2. API Deployment: The models can be deployed as APIs using Flask or FastAPI for real-time predictions.

Currently, no deployment has been performed, but the pipeline is structured to allow easy integration with deployment frameworks.

This README.md template provides a clear and concise explanation of your project’s objectives, methodology, and implementation. It also highlights key areas such as model selection, evaluation, and how the pipeline is structured for ease of use and future experimentation.