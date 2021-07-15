# Import panda
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split

# Read a CSV file into a panda data frame
data = pd.read_csv('../utilities/fiver.csv')

# Instantiate a decision tree, split the data and predict using the test data
model = tree.DecisionTreeClassifier()
X = data.drop(columns='fiver')
y = data['fiver']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Log predictions and correct values
print(f"Predictions: {predictions} - Correct answers: {y_test.values}")
