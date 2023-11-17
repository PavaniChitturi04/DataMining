from scipy.io import arff

import pandas as pd

from mlxtend.frequent_patterns import apriori, association_rules

 

# Load the ARFF file

data = arff.loadarff('C:/Users/Srinivasa Rao/OneDrive/Documents/DM/contact-lens.arff')

df = pd.DataFrame(data[0])

 

# Convert the nominal attributes to strings

for col in df.columns:

    if pd.api.types.is_categorical_dtype(df[col]):

        df[col] = df[col].str.decode('utf-8')

 

# Convert the dataset into a one-hot encoded format

oht = pd.get_dummies(df.iloc[:, :-1], columns=df.columns[:-1], prefix='', prefix_sep='')

 

# Find frequent itemsets using the Apriori algorithm

min_support = 0.2  # Minimum support threshold (adjust as needed)

frequent_itemsets = apriori(oht, min_support=min_support, use_colnames=True)

 

# Display frequent itemsets

print("Frequent Itemsets:")

print(frequent_itemsets)

 

# Find association rules

min_confidence = 0.7  # Minimum confidence threshold (adjust as needed)

association_rules_df = association_rules(frequent_itemsets, metric="lift", min_threshold=min_confidence)

 

# Display association rules

print("\nAssociation Rules:")

print(association_rules_df)