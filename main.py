import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("Titanic-Dataset.csv")

# Remove unnecessary columns
data = data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

# Convert text to numbers
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

# Fill missing values
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

# Convert Embarked to numbers
data["Embarked"] = data["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# Split features and target
X = data.drop("Survived", axis=1)
y = data["Survived"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

from sklearn.metrics import accuracy_score, confusion_matrix

# Make predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# Confusion matrix
cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)
# Visualization: survival by gender
survival_by_sex = data.groupby("Sex")["Survived"].mean()

survival_by_sex.plot(kind="bar")

plt.title("Survival Rate by Gender")
plt.xlabel("Sex (0 = male, 1 = female)")
plt.ylabel("Survival Rate")

plt.show()

