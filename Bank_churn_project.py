import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("BankData.csv")

#print first 5 rows of the database
print(df.head())

print(df.info())

#check the missing values
print(df.isnull().sum())

#remove non informative columns
df = df.drop(["CustomerId", "Surname"], axis=1)
print(df.head())

#encode categorical variables
df = pd.get_dummies(
    df,
    columns=["Geography", "Gender"],
    drop_first=True
)

print(df.head())

#Balance to Salary ratio
df["Balance_to_Salary"] = (
    df["Balance"] /
    (df["EstimatedSalary"] + 1)
)

#Product Density
df["Product_Density"] = (
    df["NumOfProducts"] /
    (df["Tenure"] + 1)
)

#Engagement Product Interaction
df["Engagement_Product"] = (
    df["IsActiveMember"] *
    df["NumOfProducts"]
)

#Age Tenure Interaction
df["Age_Tenure"] = (
    df["Age"] *
    df["Tenure"]
)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop("Exited", axis=1)
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Logi
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    solver="lbfgs",
    max_iter=2000,
    random_state=42
)
model.fit(X_train_scaled, y_train)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

#Random Forest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

#Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)

#XGBoost
from xgboost import XGBClassifier

xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)

#Model Evaluation
y_pred = model.predict(X_test_scaled)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

#Overall Correctness
print("Accuracy:", accuracy_score(y_test, y_pred))

#Control false churn predictions
print("Precision:", precision_score(y_test, y_pred))

#Recall
print("Recall:", recall_score(y_test, y_pred))

#F1 Score
print("F1 Score:", f1_score(y_test, y_pred))

#ROC AUC Score
print("ROC AUC Score:", roc_auc_score(y_test, y_pred))

# Simple plot test for the normal file runner
plt.figure()
plt.plot([1, 2, 3], [4, 5, 6], marker='o')
plt.title("Plot Test")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.grid(True)
print("Opening the test plot window now...")
plt.show(block=True)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(by="Importance", ascending=False)

print(importance)


import shap
import matplotlib.pyplot as plt

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)

plt.show()


from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

PartialDependenceDisplay.from_estimator(
    rf_model,
    X_test,
    ["Age"]
)

plt.show()

# Prediction output: churn probability and binary churn flag
churn_prob = model.predict_proba(X_test_scaled)[:, 1]
threshold = 0.5
churn_flag = (churn_prob >= threshold).astype(int)

prediction_results = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "ChurnProbability": churn_prob,
    "ChurnFlag": churn_flag
})

print("Prediction output (first 10 rows):")
print(prediction_results.head(10))

prediction_results.to_csv("churn_predictions.csv", index=False)
print("Saved churn_predictions.csv")








