import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

columns = [
    'Season', 'Age', 'Childish_Diseases', 'Accident_Trauma', 
    'Surgical_Intervention', 'High_Fevers', 'Alcohol_Frequency', 
    'Smoking_Habit', 'Sitting_Hours', 'Output'
]

# Load from txt file
df = pd.read_csv('./dataset/fertility_Diagnosis.txt', 
                 header=None, 
                 names=columns)


# Initial Exploration 
df.info()
'''
- Output: 
RangeIndex: 100 entries, 0 to 99
Data columns (total 10 columns):
#   Column                 Non-Null Count  Dtype  
---  ------                 --------------  -----  
0   Season                 100 non-null    float64
1   Age                    100 non-null    float64
2   Childish_Diseases      100 non-null    int64  
3   Accident_Trauma        100 non-null    int64  
4   Surgical_Intervention  100 non-null    int64  
5   High_Fevers            100 non-null    int64  
6   Alcohol_Frequency      100 non-null    float64
7   Smoking_Habit          100 non-null    int64  
8   Sitting_Hours          100 non-null    float64
9   Output                 100 non-null    object 
dtypes: float64(4), int64(5), object(1)
memory usage: 7.9+ KB

- Dataset Overview:
Total Entries: 100 samples
Total Columns: 10

Numerical Columns (float64):
float_columns = [
    'Season', 
    'Age', 
    'Alcohol_Frequency', 
    'Sitting_Hours'
]

Season: Scaled representation of seasons (-1 to 1)
- season_mapping = {
        -1: 'Winter',
        -0.33: 'Spring',
        0.33: 'Summer',
        1: 'Fall'
    }
Age: Normalized age (0 to 1)
- Original range: 18-36 years 
- Scaled to: 0-1
    0 represents youngest (18)
    1 represents oldest (36)
Alcohol_Frequency: Scaled alcohol consumption (0 to 1)
Sitting_Hours: Normalized sitting hours (0 to 1)

Binary/Categorical Numerical Columns (int64):
binary_columns = [
    'Childish_Diseases',      # 0: No, 1: Yes
    'Accident_Trauma',        # 0: No, 1: Yes
    'Surgical_Intervention',  # 0: No, 1: Yes
    'High_Fevers',            # -1, 0, 1 (fever timeline)
    'Smoking_Habit'           # -1: Never, 0: Occasional, 1: Daily
]

- binary_health_columns = [
        'Childish_Diseases',
        'Accident_Trauma',
        'Surgical_Intervention'
    ]
    # 0: No event/condition
    # 1: Event/condition occurred

- high_fevers_mapping = {
        -1: 'Less than 3 months ago',
        0: 'More than 3 months ago',
        1: 'No fever'
    }

- smoking_mapping = {
        -1: 'Never',
        0: 'Occasional',
        1: 'Daily'
    }

Categorical Column:
categorical_column = ['Output']  # N: Normal, O: Altered
output_mapping = {
    'N': 'Normal Sperm Quality',
    'O': 'Altered Sperm Quality'
}
'''
print("Dataset Statistics:")
print(df.describe())
'''
        Season         Age  Childish_Diseases  Accident_Trauma  Surgical_Intervention  High_Fevers  Alcohol_Frequency  Smoking_Habit  Sitting_Hours
count  100.000000  100.000000         100.000000       100.000000             100.000000   100.000000         100.000000     100.000000     100.000000
mean    -0.078900    0.669000           0.870000         0.440000               0.510000     0.190000           0.832000      -0.350000       0.406800
std      0.796725    0.121319           0.337998         0.498888               0.502418     0.580752           0.167501       0.808728       0.186395
min     -1.000000    0.500000           0.000000         0.000000               0.000000    -1.000000           0.200000      -1.000000       0.060000
25%     -1.000000    0.560000           1.000000         0.000000               0.000000     0.000000           0.800000      -1.000000       0.250000
50%     -0.330000    0.670000           1.000000         0.000000               1.000000     0.000000           0.800000      -1.000000       0.380000
75%      1.000000    0.750000           1.000000         1.000000               1.000000     1.000000           1.000000       0.000000       0.500000
max      1.000000    1.000000           1.000000         1.000000               1.000000     1.000000           1.000000       1.000000       1.000000
'''

print("\nOutput Distribution:")
print(df['Output'].value_counts(normalize=True))
'''
Output Distribution:
Output
N    0.88
O    0.12
Name: proportion, dtype: float64

- Understanding: Highly imbalanced data, Majority of normal sperm quality.
- Model may get imbalanced
'''

# Correlation Analysis
print("\nCorrelation Matrix:")
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
print(correlation_matrix)
'''
Correlation Matrix:
                         Season       Age  Childish_Diseases  Accident_Trauma  ...  High_Fevers  Alcohol_Frequency  Smoking_Habit  Sitting_Hours
Season                 1.000000  0.065410          -0.176509        -0.096274  ...    -0.221818          -0.041290      -0.028085      -0.019021
Age                    0.065410  1.000000           0.080551         0.215958  ...     0.120284          -0.247940       0.072581      -0.442452
Childish_Diseases     -0.176509  0.080551           1.000000         0.162936  ...     0.075645           0.038538       0.090535      -0.147761
Accident_Trauma       -0.096274  0.215958           0.162936         1.000000  ...    -0.082278          -0.242722       0.110157       0.013122
Surgical_Intervention -0.006210  0.271945          -0.140972         0.103166  ...    -0.231598          -0.075858      -0.053448      -0.192726
High_Fevers           -0.221818  0.120284           0.075645        -0.082278  ...     1.000000          -0.000831      -0.007527      -0.151091
Alcohol_Frequency     -0.041290 -0.247940           0.038538        -0.242722  ...    -0.000831           1.000000      -0.184926       0.111371
Smoking_Habit         -0.028085  0.072581           0.090535         0.110157  ...    -0.007527          -0.184926       1.000000      -0.106007
Sitting_Hours         -0.019021 -0.442452          -0.147761         0.013122  ...    -0.151091           0.111371      -0.106007       1.000000

'''

# Output Distribution
plt.figure(figsize=(10, 6))
df['Output'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Sperm Quality Distribution')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()