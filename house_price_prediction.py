print("Program Started")

from sklearn.datasets import fetch_california_housing
import pandas as pd

housing = fetch_california_housing()

df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

df["Price"] = housing.target

print(df.head())

from sklearn.model_selection import train_test_split

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

print("Model Trained Successfully!")
predictions = model.predict(X_test)
for i in range(5):
    print(predictions[i])
print("\nActual Price    Predicted Price")

for i in range(5):
    print(y_test.iloc[i], "      ", predictions[i])
    from sklearn.metrics import r2_score

score = r2_score(y_test, predictions)

print("\nR2 Score:", score)
import pickle

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model Saved Successfully!")