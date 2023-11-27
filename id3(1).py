# Import necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

# Create a DataFrame with the provided data
data = pd.read_csv(r"C:\Users\Srinivasa Rao\Downloads\employee(id3).csv")

df = pd.DataFrame(data)

# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(df[['age', 'salary']])
y = df['performance']

# Create and fit the Decision Tree model
model = DecisionTreeClassifier(criterion='entropy')  # ID3 uses entropy as the criterion
model.fit(X, y)

# Display the decision tree rules
tree_rules = export_text(model, feature_names=list(X.columns))
print(tree_rules)