# demonstrate of preprocessing on dataset labour.arff using python
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder

# Load ARFF file
data, meta = arff.loadarff(r"C:\Users\Srinivasa Rao\OneDrive\Documents\DM\labor.arff")

# Convert ARFF data to DataFrame
df = pd.DataFrame(data)

# Replace '?' with NaN to handle missing values
df.replace('?', pd.NA, inplace=True)

# Encode categorical variables using LabelEncoder
categorical_columns = ['cost-of-living-adjustment', 'pension', 'education-allowance', 'vacation', 'longterm-disability-assistance', 'contribution-to-dental-plan']
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le

# Split the dataset into features (X) and target (y)
X = df.drop('vacation', axis=1)
y = df['vacation']

# Your dataset is now preprocessed and ready for data mining tasks.
print(X)
print(y)