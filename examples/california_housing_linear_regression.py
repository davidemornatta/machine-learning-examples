# Import datasets, regression classifier
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

california_housing = datasets.fetch_california_housing()
data = california_housing.data
target = california_housing.target

# Split data into 80% train and 20% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, shuffle=False)

# Create a classifier: a linear regression classifier
model = LinearRegression()

# Fit the model with the train data
model.fit(X_train, y_train)

# Predict the value of the price on the test subset
predicted = model.predict(X_test)

# Print model score and first 10 correct value - predicted value
print(f"Model score: {model.score(X_test, y_test)}")

for index, correct_value, prediction in zip(range(1, 11), y_test, predicted):
    print(f"Correct value: {correct_value} - Prediction: {prediction}")

