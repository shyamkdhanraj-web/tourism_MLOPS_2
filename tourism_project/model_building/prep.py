import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("tourism_project/data/tourism.csv")

# Drop redundant identifier and index columns
cols_to_drop = ["Unnamed: 0", "CustomerID"]
df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True, errors="ignore")

# Clean inconsistent categorical values to prevent downstream encoding issues
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].replace({'Fe Male': 'Female'})

if 'MaritalStatus' in df.columns:
    df['MaritalStatus'] = df['MaritalStatus'].replace({'Unmarried': 'Single'})

# Define features and target variable
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# stratify=y keeps the purchase ratio consistent across train and test splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save train/test splits
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("Target variable: ProdTaken")
print("Number of features:", X.shape[1])
