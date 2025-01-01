#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sqlite3
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline


# In[2]:


conn = sqlite3.connect('data/agri.db')
cursor = conn.cursor()

# List all table names in the database
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()


# In[3]:


# Query the data from the 'farm_data' table and load it into a pandas DataFrame
df = pd.read_sql_query("SELECT * FROM farm_data", conn)

cursor.close()
conn.close()


# In[4]:


'''Now we'll identify patterns, correlations, and potential causal relationships
between various variables. Because Nutrients are in object (data type), we did 
a quick conversion to numeric.
'''

def prepare_data(df):
    """
    Prepare the data by converting nutrient columns to numeric and handling null values
    """
    df_clean = df.copy()
    
    # Convert nutrient columns from object to numeric
    nutrient_cols = ['Nutrient N Sensor (ppm)', 'Nutrient P Sensor (ppm)', 'Nutrient K Sensor (ppm)']
    for col in nutrient_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    return df_clean


# In[5]:


# Let's start wth the data type of the three Nutrient columns

def convert_nutrient_columns(df):
    """
    Convert nutrient columns from object to numeric type and display conversion summary
    
    Parameters:
    - df: Original DataFrame
    
    Returns:
    - DataFrame with converted nutrient columns
    - Dictionary containing conversion summary
    """
    # Create a copy of the dataframe
    df_clean = df.copy()
    
    # List of nutrient columns
    nutrient_cols = [
        'Nutrient N Sensor (ppm)',
        'Nutrient P Sensor (ppm)',
        'Nutrient K Sensor (ppm)'
    ]
    
    # Dictionary to store conversion summary
    conversion_summary = {}
    
    for col in nutrient_cols:
        # Store original info
        original_nulls = df[col].isnull().sum()
        
        # Convert to numeric
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Store conversion summary
        conversion_summary[col] = {
            'original_dtype': df[col].dtype,
            'new_dtype': df_clean[col].dtype,
            'original_nulls': original_nulls,
            'new_nulls': df_clean[col].isnull().sum(),
            'additional_nulls': df_clean[col].isnull().sum() - original_nulls,
            'min_value': df_clean[col].min(),
            'max_value': df_clean[col].max(),
            'mean_value': df_clean[col].mean()
        }
    
    return df_clean, conversion_summary

# To use the function:
df_clean, summary = convert_nutrient_columns(df)


# In[6]:


# Next, we standardise the text casing of Plant Type

def standardize_plant_type(column):
    """
    Standardize Plant Type values to consistent case format and display changes
    
    Parameters:
    - column: Series containing Plant Type values
    
    Returns:
    - Standardized Series
    """
    # Get original value counts
    original_values = column.value_counts()
    
    # Standardize the values (convert to title case)
    standardized_column = column.str.title()
    
    # Get new value counts
    new_values = standardized_column.value_counts()
    
    return standardized_column

# Apply the standardization to the Plant Type column in df_clean
df_clean['Plant Type'] = standardize_plant_type(df_clean['Plant Type'])


# In[7]:


# Now, we do the same for Plant Stage

def standardize_plant_stage(column):
    """
    Standardize Plant Stage values to consistent case format and display changes
    
    Parameters:
    - column: Series containing Plant Stage values
    
    Returns:
    - Standardized Series
    """
    # Get original value counts
    original_values = column.value_counts()
    
    # Standardize the values (convert to title case)
    standardized_column = column.str.title()
    
    # Get new value counts
    new_values = standardized_column.value_counts()
    
    return standardized_column

# Apply the standardization to the Plant Stage column in df_clean
df_clean['Plant Stage'] = standardize_plant_stage(df_clean['Plant Stage'])


# In[8]:


# Now, we convert the variables that are integers to float
# It's just to be safe, to avoid any unintended integer division issues.

def convert_sensor_values(df):
    """
    Convert CO2 and O2 sensor values from int64 to float64 and display conversion summary
    
    Parameters:
    - df: DataFrame containing sensor data
    
    Returns:
    - DataFrame with converted sensor columns
    - Dictionary containing conversion summary
    """
    # Create a copy of the dataframe
    df_clean = df.copy()
    
    # List of columns to convert
    sensor_cols = ['CO2 Sensor (ppm)', 'O2 Sensor (ppm)']
    
    # Dictionary to store conversion summary
    conversion_summary = {}
    
    for col in sensor_cols:
        # Store original info
        conversion_summary[col] = {
            'original_dtype': df[col].dtype,
            'min_value': df[col].min(),
            'max_value': df[col].max(),
            'mean_value': df[col].mean()
        }
        
        # Convert to float
        df_clean[col] = df_clean[col].astype(float)
        
        # Store new info
        conversion_summary[col].update({
            'new_dtype': df_clean[col].dtype,
            'values_changed': not (df_clean[col] == df[col]).all()
        })
    
    return df_clean, conversion_summary

# Apply this with rest of variables in df_clean
df_clean, sensor_summary = convert_sensor_values(df_clean)


# In[9]:


# 1. First, create a subset of complete cases for humidity analysis
humidity_complete = df_clean.dropna(subset=['Humidity Sensor (%)'])

# 2. For the main dataset, use multiple imputation considering the correlations
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Create an imputer that uses the correlations we observed
imputer = IterativeImputer(
    random_state=42,
    max_iter=10,
    initial_strategy='mean',
    min_value=0,  # Humidity can't be negative
    max_value=100  # Humidity can't be over 100%
)

# Select columns for imputation based on strongest correlations
columns_for_imputation = [
    'Temperature Sensor (°C)',
    'Humidity Sensor (%)',
    'Light Intensity Sensor (lux)',
    'CO2 Sensor (ppm)',
    'Nutrient K Sensor (ppm)',
    'Nutrient P Sensor (ppm)'
]

# Perform imputation
df_imputed = df_clean.copy()
df_imputed[columns_for_imputation] = imputer.fit_transform(df_clean[columns_for_imputation])


# In[10]:


# Back up original data just in case
df_original = df.copy()

# Overwrite df_clean with our imputed data
df_clean = df_imputed.copy()


# In[11]:


# We've left with missing values with two more variables — Nutrient K and Water Level

# Set up imputer for Nutrient N and Water Level
imputer = IterativeImputer(
    random_state=42,
    max_iter=10,
    initial_strategy='mean',
    min_value=0  # Both sensors can't have negative values
)

# Select relevant columns for imputation
columns_for_imputation = [
    'Nutrient N Sensor (ppm)',
    'Nutrient P Sensor (ppm)',
    'Nutrient K Sensor (ppm)',
    'Water Level Sensor (mm)',
    'Temperature Sensor (°C)',
    'Humidity Sensor (%)',
    'Light Intensity Sensor (lux)',
    'CO2 Sensor (ppm)'
]

# Perform imputation
df_clean[columns_for_imputation] = imputer.fit_transform(df_clean[columns_for_imputation])


# In[13]:


# Define the columns to handle
def handle_outliers(df, columns, method='cap', multiplier=1.5, minimum_bounds=None):
    """
    Handle outliers in specified columns using the IQR method.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    columns : list
        List of column names to handle outliers
    method : str, optional (default='cap')
        'cap': Cap outliers at the IQR bounds
        'remove': Remove rows with outliers
    multiplier : float, optional (default=1.5)
        IQR multiplier for calculating bounds
    minimum_bounds : dict, optional
        Dictionary specifying minimum allowed values for specific columns
        e.g., {'Light Intensity Sensor (lux)': 0}
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with handled outliers
    dict
        Dictionary containing original and new value counts for each column
    """

    # Create a copy of the DataFrame
    df_outliers = df_clean.copy()
    
    # Initialize minimum_bounds if None
    if minimum_bounds is None:
        minimum_bounds = {}
    
    # Dictionary to store statistics
    stats = {}
    
    for col in columns:
        # Calculate IQR bounds
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        # Adjust lower bound if minimum value is specified
        if col in minimum_bounds:
            lower_bound = max(lower_bound, minimum_bounds[col])
        
        # Store original count of outliers
        original_outliers = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
        
        if method == 'cap':
            # Cap the outliers at the bounds
            df_outliers[col] = df_outliers[col].clip(lower=lower_bound, upper=upper_bound)
            new_outliers = 0
        else:  # method == 'remove'
            # Remove rows with outliers
            mask = (df_outliers[col] >= lower_bound) & (df_outliers[col] <= upper_bound)
            df_outliers = df_outliers[mask]
            new_outliers = len(df_clean) - len(df_outliers)
        
        # Store statistics
        stats[col] = {
            'original_outliers': original_outliers,
            'new_outliers': new_outliers,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'percentage_affected': (original_outliers / len(df)) * 100
        }
    
    return df_outliers, stats

# Define minimum bounds for sensors where negative values are impossible
minimum_bounds = {
    'Light Intensity Sensor (lux)': 0,
    'Temperature Sensor (°C)': -10  # Optional: you might want to set a reasonable minimum temperature
}

# Define the columns to handle
columns_to_handle = ['Temperature Sensor (°C)', 'Light Intensity Sensor (lux)', 'EC Sensor (dS/m)', 'Water Level Sensor (mm)']

# Apply the function with minimum bounds
df_outliers_handled, outlier_stats = handle_outliers(
    df_clean, 
    columns_to_handle, 
    method='cap',
    minimum_bounds=minimum_bounds
)


# In[14]:


def handle_light_intensity_outliers(df_clean, column_name='Light Intensity Sensor (lux)'):
    """
    Handle outliers in light intensity sensor data using the IQR method.
    Ensures no negative values are present in the final output.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    column_name (str): Name of the column containing light intensity values
    
    Returns:
    pandas.DataFrame: DataFrame with outliers handled
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_processed = df_clean.copy()
    
    # Calculate Q1, Q3, and IQR
    Q1 = df_processed[column_name].quantile(0.25)
    Q3 = df_processed[column_name].quantile(0.75)
    IQR = Q3 - Q1
    
    # Calculate bounds
    lower_bound = max(0, Q1 - 1.5 * IQR)  # Ensure lower bound is not negative
    upper_bound = Q3 + 1.5 * IQR
    
    # Print diagnostic information
    print(f"Original data summary:")
    print(f"Number of records: {len(df_processed)}")
    print(f"Q1: {Q1:.2f}")
    print(f"Q3: {Q3:.2f}")
    print(f"IQR: {IQR:.2f}")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")
    
    # Identify outliers
    outliers = df_processed[
        (df_processed[column_name] < lower_bound) | 
        (df_processed[column_name] > upper_bound)
    ]
    
    # Handle outliers by capping
    df_processed[column_name] = df_processed[column_name].clip(lower=lower_bound, upper=upper_bound)
    
    return df_processed

# Apply outlier handling to the cleaned dataset
df_clean = handle_light_intensity_outliers(df_clean)


# In[15]:


# Check and handle duplicates
duplicate_count = df_outliers_handled.duplicated().sum()
if duplicate_count > 0:
    df_no_duplicate = df_outliers_handled.drop_duplicates()


# In[16]:


# Create a copy of the dataframe to preserve the original
df_scaled = df_no_duplicate.copy()

# Variables for StandardScaler
standard_columns = [
    'Temperature Sensor (°C)',
    'Light Intensity Sensor (lux)',
    'CO2 Sensor (ppm)',
    'O2 Sensor (ppm)',
    'EC Sensor (dS/m)',
    'Nutrient N Sensor (ppm)',
    'Nutrient P Sensor (ppm)',
    'Nutrient K Sensor (ppm)'
]

# Variables for MinMaxScaler
minmax_columns = [
    'Humidity Sensor (%)',
    'pH Sensor',
    'Water Level Sensor (mm)'
]

# Apply StandardScaler
std_scaler = StandardScaler()
df_scaled[standard_columns] = std_scaler.fit_transform(df_no_duplicate[standard_columns])

# Apply MinMaxScaler
mm_scaler = MinMaxScaler()
df_scaled[minmax_columns] = mm_scaler.fit_transform(df_no_duplicate[minmax_columns])


# In[17]:


# Create copies for each task to avoid modifications
df_temperature = df_scaled.copy()
df_plant_class = df_scaled.copy()

# === TEMPERATURE PREDICTION TASK ===
# Create environmental interactions
df_temperature['humidity_light_interaction'] = df_temperature['Humidity Sensor (%)'] * df_temperature['Light Intensity Sensor (lux)']
df_temperature['co2_light_interaction'] = df_temperature['CO2 Sensor (ppm)'] * df_temperature['Light Intensity Sensor (lux)']
df_temperature['water_humidity_interaction'] = df_temperature['Water Level Sensor (mm)'] * df_temperature['Humidity Sensor (%)']

# One-hot encode location
location_encoded = pd.get_dummies(df_temperature['System Location Code'], prefix='location')
df_temperature = pd.concat([df_temperature, location_encoded], axis=1)

# Create zone-specific environmental averages
df_temperature['zone_humidity_mean'] = df_temperature.groupby('System Location Code')['Humidity Sensor (%)'].transform('mean')

# Drop the original System Location Code column since we now have one-hot encoded versions
df_temperature.drop('System Location Code', axis=1, inplace=True)

# === PLANT TYPE-STAGE CLASSIFICATION TASK ===
# Create target variable
df_plant_class['plant_type_stage'] = df_plant_class['Plant Type'] + '_' + df_plant_class['Plant Stage']

# Create NPK ratio
df_plant_class['npk_ratio'] = (df_plant_class['Nutrient N Sensor (ppm)'] + 
                              df_plant_class['Nutrient P Sensor (ppm)'] + 
                              df_plant_class['Nutrient K Sensor (ppm)']) / 3

# Create growth environment score
df_plant_class['growth_environment_score'] = (
    df_plant_class['Light Intensity Sensor (lux)'] +
    df_plant_class['CO2 Sensor (ppm)'] +
    df_plant_class['O2 Sensor (ppm)']
) / 3

# Create water conditions score
df_plant_class['water_conditions_score'] = (
    df_plant_class['EC Sensor (dS/m)'] +
    df_plant_class['pH Sensor'] +
    df_plant_class['Water Level Sensor (mm)']
) / 3

# Create crop rotation feature
df_plant_class['crop_rotation'] = (df_plant_class['Plant Type'] != df_plant_class['Previous Cycle Plant Type']).astype(int)

# One-hot encode location for classification task as well
location_encoded = pd.get_dummies(df_plant_class['System Location Code'], prefix='location')
df_plant_class = pd.concat([df_plant_class, location_encoded], axis=1)

# Drop original categorical columns that we've encoded or don't need
df_plant_class.drop(['System Location Code', 'Previous Cycle Plant Type'], axis=1, inplace=True)


# In[18]:


# Basic statistics for new features in Temperature Prediction dataset
new_temp_features = ['humidity_light_interaction', 'co2_light_interaction', 
                     'water_humidity_interaction', 'zone_humidity_mean']

# Basic statistics for new features in Plant Classification dataset
new_plant_features = ['npk_ratio', 'growth_environment_score', 
                      'water_conditions_score', 'crop_rotation']

# For Plant Classification, let's look at feature importance by plant type-stage
# First, let's encode plant_type_stage
le = LabelEncoder()
df_plant_class['plant_type_stage_encoded'] = le.fit_transform(df_plant_class['plant_type_stage'])

# Calculate the correlation between zone locations and temperature
zone_temp_corr = df_temperature[[col for col in df_temperature.columns if 'location_Zone' in col] + ['Temperature Sensor (°C)']].corr()['Temperature Sensor (°C)'].sort_values(ascending=False)


# In[19]:


'''FOR TEMPERATURE PREDICTION (REGRESSION)'''

# Define features and target
X = df_temperature[new_temp_features]  # Using features list from earlier
y = df_temperature['Temperature Sensor (°C)']

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# In[20]:


def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
# 1. Linear Regression (Most interpretable)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_val)
evaluate_model(y_val, lr_pred, "Linear Regression")

# 2. Ridge Regression (L2 regularization)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_val)
evaluate_model(y_val, ridge_pred, "Ridge Regression")

# 3. Random Forest (More complex but still interpretable)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_val)
evaluate_model(y_val, rf_pred, "Random Forest")


# In[23]:


def plot_feature_importance(model, model_name):
    
    # Get feature importance based on model type
    if isinstance(model, (LinearRegression, Ridge, Lasso)):
        # For linear models, use absolute coefficients
        importance = np.abs(model.coef_)
    elif isinstance(model, RandomForestRegressor):
        # For Random Forest, use feature_importances_
        importance = model.feature_importances_
    else:
        raise ValueError("Unsupported model type")
    
    # Create DataFrame of features and their importance
    feat_importance = pd.DataFrame({
        'feature': ['humidity_light_interaction', 'co2_light_interaction', 
                   'water_humidity_interaction', 'zone_humidity_mean'],
        'importance': importance
    })
    
    # Sort by importance
    feat_importance = feat_importance.sort_values('importance', ascending=False)
    
    return feat_importance


# In[24]:


# Function for model evaluation
def evaluate_regression_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return mae, mse, rmse, r2

# Try different regularization strengths
alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# Ridge Regression
ridge_results = []
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_val)
    metrics = evaluate_regression_model(y_val, y_pred, f"Ridge (alpha={alpha})")
    ridge_results.append((alpha, *metrics))

# Lasso Regression
lasso_results = []
for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_val)
    metrics = evaluate_regression_model(y_val, y_pred, f"Lasso (alpha={alpha})")
    lasso_results.append((alpha, *metrics))


# In[25]:


# GridSearchCV for Ridge with minimal parameters and single job
ridge_params = {
    'alpha': [0.1, 1.0, 10.0]  # Just 3 options
}

grid_ridge = GridSearchCV(
    Ridge(),
    ridge_params,
    cv=3,  # Reduced from 5 to 3
    scoring='neg_mean_squared_error',
    n_jobs=1,  # Single job - no parallel processing
    verbose=0
)
grid_ridge.fit(X_train, y_train)


# RandomizedSearchCV for Random Forest with minimal parameters
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

random_rf = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    rf_params,
    n_iter=5,  # Only 5 iterations
    cv=3,  # Reduced from 5 to 3
    scoring='neg_mean_squared_error',
    n_jobs=1,  # Single job - no parallel processing
    verbose=0,
    random_state=42
)
random_rf.fit(X_train, y_train)


# In[26]:


# Get best models
best_ridge = grid_ridge.best_estimator_
best_rf = random_rf.best_estimator_

# Function for final evaluation
def final_model_evaluation(model, model_name):
    # Predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    evaluate_regression_model(y_train, train_pred, "Training")
    evaluate_regression_model(y_val, val_pred, "Validation")
    evaluate_regression_model(y_test, test_pred, "Test")

# Evaluate best models
final_model_evaluation(best_ridge, "Best Ridge Model")
final_model_evaluation(best_rf, "Best Random Forest Model")


# In[27]:


'''PLANT CLASSIFICATION (CATEGORICAL)'''

# Define features (X) and y (target)
X = df_plant_class[[
    'npk_ratio', 
    'growth_environment_score', 
    'water_conditions_score', 
    'crop_rotation',
    'location_Zone_A', 
    'location_Zone_B', 
    'location_Zone_C', 
    'location_Zone_D', 
    'location_Zone_E', 
    'location_Zone_F', 
    'location_Zone_G'
]]
y = df_plant_class['plant_type_stage']  # Using string labels instead of encoded values for better interpretability

# Perform the splits
from sklearn.model_selection import train_test_split

# First split: separate test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.15,
    stratify=y,
    random_state=42
)

# Second split: separate train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.176,  # 0.15/0.85 to get 15% of original data
    stratify=y_temp,
    random_state=42
)

# Optional: Save the initial split sizes for later reference
split_sizes = {
    'train': X_train.shape[0],
    'val': X_val.shape[0],
    'test': X_test.shape[0],
    'total': len(X)
}


# In[28]:


# Train Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

# Fit the model
rf_model.fit(X_train, y_train)

# Make predictions on validation set
y_val_pred = rf_model.predict(X_val)

# Plot feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Plot confusion matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_val, y_val_pred)

# Save baseline scores for comparison
baseline_scores = {
    'model': 'Random Forest Baseline',
    'accuracy': rf_model.score(X_val, y_val),
    'feature_importance': feature_importance
}


# In[29]:


# Create interaction features
def create_interaction_features(X):
    X = X.copy()
    X['npk_growth_interaction'] = X['npk_ratio'] * X['growth_environment_score']
    X['npk_water_interaction'] = X['npk_ratio'] * X['water_conditions_score']
    X['growth_water_interaction'] = X['growth_environment_score'] * X['water_conditions_score']
    return X

# Create pipeline
pipeline = Pipeline([
    ('interaction_features', FunctionTransformer(create_interaction_features)),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    ))
])

# Fit pipeline
pipeline.fit(X_train, y_train)

# Evaluate
y_val_pred_improved = pipeline.predict(X_val)


# In[30]:


# Define simplified parameter space
param_dist = {
    'n_estimators': [100, 200],  # Just two options
    'max_depth': [10, None],     # Just two options
    'min_samples_split': [2, 5],  # Just two options
    'min_samples_leaf': [1, 2]    # Just two options
}

# Random Search with minimal settings
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=5,          # Reduced iterations
    cv=3,              # Reduced from 5 to 3
    scoring='accuracy',
    n_jobs=1,          # Single job - no parallel processing
    verbose=0,         # Reduce output
    random_state=42
)

random_search.fit(X_train, y_train)

# Get best model for final evaluation
best_model = random_search.best_estimator_


# In[31]:


def final_model_evaluation(model, X_test, y_test, feature_columns):
    # Get predictions on test set
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)
    
    # Overall Accuracy
    accuracy = accuracy_score(y_test, y_test_pred)

    # Feature Importance (if using Random Forest or similar)
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)

    # Per-class Metrics
    per_class_metrics = pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True)).T

    # Save results (optional)
    results = {
        'accuracy': accuracy,
        'classification_report': classification_report(y_test, y_test_pred),
        'confusion_matrix': cm,
        'feature_importance': feature_importance if hasattr(model, 'feature_importances_') else None
    }
    
    return results

# Run the final evaluation
final_results = final_model_evaluation(
    model=best_model,  # Use your best model from hyperparameter tuning
    X_test=X_test,
    y_test=y_test,
    feature_columns=feature_columns
)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




