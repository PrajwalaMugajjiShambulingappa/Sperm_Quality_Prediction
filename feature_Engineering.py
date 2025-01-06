import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold

# Define the column names for the dataset
columns = [
    'Season', 'Age', 'Childish_Diseases', 'Accident_Trauma',
    'Surgical_Intervention', 'High_Fevers', 'Alcohol_Frequency',
    'Smoking_Habit', 'Sitting_Hours', 'Output'
]

# Load the fertility dataset from a .txt file
df = pd.read_csv('./dataset/fertility_Diagnosis.txt', header=None, names=columns)

# Encode the dependent variable 'Output' into numerical format
le = LabelEncoder()
df['Output_Encoded'] = le.fit_transform(df['Output'])

# Standardize/scale numeric features to have a mean of 0 and standard deviation of 1
scaler = StandardScaler()
scaled_features = ['Age', 'Alcohol_Frequency', 'Sitting_Hours']
df[scaled_features] = scaler.fit_transform(df[scaled_features])

# Create interaction features by combining existing features
# Multiplicative interaction between 'Age' and 'Sitting_Hours'
df['Age_Sitting_Interaction'] = df['Age'] * df['Sitting_Hours']
# Multiplicative interaction between 'Smoking_Habit' and 'Alcohol_Frequency'
df['Smoking_Alcohol_Interaction'] = df['Smoking_Habit'] * df['Alcohol_Frequency']
# Aggregate health risk score using binary variables
df['Health_Risk_Score'] = (
    df['Childish_Diseases'] +
    df['Accident_Trauma'] +
    df['Surgical_Intervention'] +
    abs(df['High_Fevers'])  # Convert High_Fevers to absolute values to handle negative values
)

# Generate polynomial features up to degree 2 for selected features
poly = PolynomialFeatures(degree=2, include_bias=False)
# Modify the polynomial features section to avoid duplicate columns
poly_features = ['Age', 'Sitting_Hours', 'Alcohol_Frequency']
poly_matrix = poly.fit_transform(df[poly_features])
poly_feature_names = [name for name in poly.get_feature_names_out(poly_features) 
                     if name not in poly_features]  # Exclude original features
poly_df = pd.DataFrame(poly_matrix[:, len(poly_features):], columns=poly_feature_names)
df = pd.concat([df, poly_df], axis=1)


# One-hot encode the categorical 'Season' column to represent it as binary variables
df = pd.get_dummies(df, columns=['Season'], prefix='Season')

# Map Smoking_Habit to ordinal values for modeling purposes
smoking_map = {-1: 0, 0: 1, 1: 2}  # Assign values based on the severity or levels
df['Smoking_Ordinal'] = df['Smoking_Habit'].map(smoking_map)

# Define a function to calculate a risk probability score based on specific conditions
def calculate_risk_probability(row):
    """
    Calculates the fertility risk probability based on certain health risk factors.
    Args:
        row (pd.Series): A single row of the DataFrame.
    Returns:
        float: The mean of the weighted risk factors for the individual.
    """
    try:
        # Get the first occurrence of each column if duplicated
        alcohol_freq = row['Alcohol_Frequency'].iloc[0] if isinstance(row['Alcohol_Frequency'], pd.Series) else row['Alcohol_Frequency']
        sitting_hours = row['Sitting_Hours'].iloc[0] if isinstance(row['Sitting_Hours'], pd.Series) else row['Sitting_Hours']
        
        # Convert risk factors to explicit boolean values
        risk_factors = [
            bool(row['Smoking_Habit'] == 1),  # Smoking habit present
            bool(alcohol_freq > 0.7),  # High alcohol consumption
            bool(sitting_hours > 0.7),  # Prolonged sitting hours
            bool(row['High_Fevers'] == -1),  # Unresolved high fevers
            bool(row['Accident_Trauma'] == 1)  # Accident or trauma history
        ]
        
        # Calculate mean of boolean values
        return float(sum(risk_factors)) / len(risk_factors)
    except Exception as e:
        print(f"Error in row: {row.to_dict()}")
        raise e
    
# Check for unexpected NaN values in key columns
print(df.isna().sum())

# Check data types of columns
print(df.dtypes)

# Verify that the key columns involved in `calculate_risk_probability` are numeric
print(df[['Smoking_Habit', 'Alcohol_Frequency', 'Sitting_Hours', 'High_Fevers', 'Accident_Trauma']].head())

# Check if any rows have unexpected values in these columns
print(df[['Smoking_Habit', 'Alcohol_Frequency', 'Sitting_Hours', 'High_Fevers', 'Accident_Trauma']].describe())

# Create scaled features separately
scaled_columns = ['Age', 'Alcohol_Frequency', 'Sitting_Hours']
df_scaled = df.copy()
df_scaled[scaled_columns] = scaler.fit_transform(df_scaled[scaled_columns])

# Use the non-scaled (original) columns in `calculate_risk_probability`
# Apply the risk probability calculation to each row in the dataset
# Select only the necessary columns for risk calculation
risk_columns = ['Smoking_Habit', 'Alcohol_Frequency', 'Sitting_Hours', 
                'High_Fevers', 'Accident_Trauma']
df['Fertility_Risk_Probability'] = df[risk_columns].apply(calculate_risk_probability, axis=1)

# Replace any NaN values in the DataFrame with zeros for consistency
df.fillna(0, inplace=True)

# Prepare independent (X) and dependent (y) variables for feature selection
X = df.drop(['Output', 'Output_Encoded'], axis=1)  # Drop the target column(s) from the features
y = df['Output_Encoded']                           # Use the encoded target variable

# Remove constant features (features with 0 variance) from the dataset
constant_filter = VarianceThreshold(threshold=0.0)
X = constant_filter.fit_transform(X)

# Use SelectKBest to select the top 8 features based on ANOVA F-test scores
selector = SelectKBest(score_func=f_classif, k=8)  # Select the top 8 features
X_new = selector.fit_transform(X, y)              # Apply feature selection
selected_feature_indices = selector.get_support(indices=True)  # Get indices of selected features

# Retrieve the names of the selected features based on their indices
selected_features = [df.columns[i] for i in selected_feature_indices]

# Create a new DataFrame with the selected features and the target variable
X_selected_df = pd.DataFrame(X_new, columns=selected_features)
processed_df = pd.concat([X_selected_df, y.reset_index(drop=True)], axis=1)

# Save the processed dataset to a CSV file
processed_df.to_csv('./dataset/processed_fertility_dataset.csv', index=False)
print("Processed dataset saved as 'processed_fertility_dataset.csv'")
