import pandas as pd, pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv("data.csv")
X = df[["Hours", "Health", "Understanding"]]
y = df["Scores"]

# check accuracy on data the model has not seen
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
check = LinearRegression().fit(Xtr, ytr)
pred = check.predict(Xte)
print("Test R2:", round(r2_score(yte, pred), 3), "| MAE:", round(mean_absolute_error(yte, pred), 2))

# final model on all data
model = LinearRegression().fit(X, y)
print("Coefficients:", dict(zip(X.columns, model.coef_.round(3))), "| Intercept:", round(model.intercept_, 3))
pickle.dump(model, open("regressor.pickle", "wb"))
print("Saved regressor.pickle")   