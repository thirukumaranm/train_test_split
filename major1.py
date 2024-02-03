# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE  # If needed for class imbalance handling

# Load the dataset
dataset_path = "your_dataset.csv"
df = pd.read_csv(dataset_path)

# Data Inspection
print("Dataset Overview:")
print(df.info())
print("\nFirst few rows of the dataset:")
print(df.head())

# Separate features and target variable (adjust accordingly)
X = df.drop("QUANTITY", axis=1)
y = df["QUANTITY"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Preprocessing and Feature Engineering Pipeline
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Data Cleaning - Handling missing values
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

# Feature Scaling - Standardization
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Handling Categorical Data - One-Hot Encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply preprocessing to the training data
X_train_preprocessed = preprocessor.fit_transform(X_train)

# Feature Engineering - Apply your feature engineering techniques here

# Handling Imbalanced Data - Example using SMOTE
smote = SMOTE(random_state=42)
X_train_preprocessed, y_train = smote.fit_resample(X_train_preprocessed, y_train)

# Save preprocessed dataset as a CSV file
preprocessed_path = "preprocessed_dataset.csv"
pd.DataFrame(X_train_preprocessed).to_csv(preprocessed_path, index=False)

# Analysis - Visualizations and summary statistics (you can add your analysis here)
# ...

print("Data Preprocessing and Feature Engineering completed. Preprocessed dataset saved at:", preprocessed_path)
