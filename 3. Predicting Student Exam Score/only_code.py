import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import  train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

df = pd.read_csv(r"C:\Users\Shivam Mourya\ML PRO\Predicting Students Exam Score\student_habits_performance.csv")

df.head()

df.info()

# dropping NAN value rows
df = df.dropna()

#droping student id column because their is no use of it and may disturb model
df = df.drop(columns = "student_id")

df.describe(include = 'object')

df.describe(include = 'object').columns

categorical_cols = ['gender', 'part_time_job', 'diet_quality', 'parental_education_level',
       'internet_quality', 'extracurricular_participation']

for col in categorical_cols:
    print(df[col].value_counts())
    print("__"*50)

df.hist(bins=20, edgecolor="black")
plt.tight_layout()
plt.show()

for col in categorical_cols:
    sns.countplot(data=df, x=col)
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation = 45)
    plt.show()
    

plt.figure(figsize=(10, 5))
sns.heatmap(df.corr(numeric_only = True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

num_features = ['age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours',
       'attendance_percentage', 'sleep_hours', 'exercise_frequency',
       'mental_health_rating']

for feature in num_features:
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x=feature, y="exam_score")
    plt.title(f"{feature} vs Exam Score")
    plt.xlabel(feature)
    plt.ylabel("Exam Score")
    plt.show()

for col in categorical_cols:
    sns.boxplot(data=df, x=col, y="exam_score")
    plt.title(f"Exam Score by {col}")
    plt.xticks(rotation =45)
    plt.show()

df.columns

features = ['study_hours_per_day','sleep_hours','attendance_percentage','mental_health_rating', 'part_time_job']
target = ['exam_score']
df_model = df[features + target].copy()

df_model

labelencoder = LabelEncoder()
df_model['part_time_job'] = labelencoder.fit_transform(df_model['part_time_job'])

X = df_model[features]
y = df_model[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)


models = {
    "LinearRegression": {
        "model": LinearRegression(),
        "params": {}
    },
    "DecisionTree": {
        "model": DecisionTreeRegressor(),
        "params": {"max_depth": [3, 5, 10], "min_samples_split": [2, 5]}
    },
    "RandomForest": {
        "model": RandomForestRegressor(),
        "params": {"n_estimators": [50, 100], "max_depth": [5, 10]}
    }
}

        

best_model =[]

for name, config in models.items():
    print(f"Training {name}")

    grid = GridSearchCV(config["model"],config["params"], cv=5, scoring="neg_mean_squared_error")
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    r2 = r2_score(y_test, y_pred)

    best_model.append({
        "model":name,
        "best_params": grid.best_params_,
        "rmse": rmse,
        "R2": r2
    })

result_df = pd.DataFrame(best_model)
result_df.sort_values(by="rmse")

import joblib

best_row = result_df.sort_values(by="rmse").iloc[1]
best_row

best_model_name = best_row["model"]
best_row

best_model_config = models[best_model_name]
best_model_config

final_model = best_model_config["model"]

final_model.fit(X, y)

joblib.dump(final_model,"best_model.pkl")


joblib.load("best_model.pkl").predict(X_test)

import streamlit as st
import numpy as np
import joblib 
import warnings
warnings.filterwarnings("ignore")

model = joblib.load("best_model.pkl")

st.title("Student Exam Score Predictor")

study_hours = st.slider("Study Hours per Day", 0.0, 12.0, 2.0)
attendance = st.slider("Attendance Percentage", 0.0, 100.0, 80.0)
mental_health = st.slider("Mental Health rating (1-10)", 1, 10, 5)
sleep_hours = st.slider("Sleep Hours per Night", 0.0, 12.0, 7.0)
part_time_job = st.selectbox("Part-Time Job", ["No", "Yes"])

ptj_encoded = 1 if part_time_job == "Yes" else 0

if st.button("Predict Exam Score"):
    input_data = np.array([[study_hours, attendence, mental_health, sleep_hours, ptj_encoded]])
    prediction = model.predict(input_data)[0]

    prediction = max(0, min(100, prediction))

    st.success(f"Predicted Exam Score: {prediction:.2f}")