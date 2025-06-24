import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Caricamento dataset
file_path = "smart_mobility_dataset.csv"
df = pd.read_csv(file_path)

# Feature engineering (come già definito in precedenza)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
df['Weekday'] = df['Timestamp'].dt.weekday
df['Is_Weekend'] = df['Weekday'].isin([5,6]).astype(int)

def get_day_period(hour):
    if 0 <= hour < 6:
        return 'Night'
    elif 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    else:
        return 'Evening'

df['Day_Period'] = df['Hour'].apply(get_day_period)
df['Bad_Weather'] = df['Weather_Condition'].isin(['Rain','Snow']).astype(int)
df['Congestion_Index'] = df['Vehicle_Count'] * df['Road_Occupancy_%']
df['Flow_Efficiency'] = df['Traffic_Speed_kmh'] / (df['Vehicle_Count'] + 1)
df['Urban_Stress'] = df['Sentiment_Score'] * df['Accident_Report']

target = 'Traffic_Condition'
features = [
    'Latitude', 'Longitude', 'Vehicle_Count', 'Traffic_Speed_kmh', 'Road_Occupancy_%',
    'Traffic_Light_State', 'Weather_Condition', 'Accident_Report',
    'Sentiment_Score', 'Ride_Sharing_Demand', 'Parking_Availability',
    'Emission_Levels_g_km', 'Energy_Consumption_L_h',
    'Hour', 'Weekday', 'Is_Weekend', 'Day_Period', 'Bad_Weather',
    'Congestion_Index', 'Flow_Efficiency', 'Urban_Stress'
]

X = df[features]
y = df[target]

# 2. Definizione colonne numeriche e categoriche
categorical = ['Traffic_Light_State', 'Weather_Condition', 'Day_Period']
numerical = [col for col in X.columns if col not in categorical]

# 3. Pipeline preprocessing con imputazione
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first'))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical),
    ('cat', categorical_pipeline, categorical)
])

# 4. Suddivisione train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Pipeline finale con modello
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 6. Ricerca iperparametri con GridSearchCV
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

# 7. Valutazione sul test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\nTest accuracy: {accuracy:.4f}")
print("\nClassification report:\n", report)
print("Confusion matrix:\n", conf_matrix)

# 8. Analisi feature importance
importances = best_model.named_steps['classifier'].feature_importances_

# Ottieni nomi feature preprocessate
feature_names_num = numerical
feature_names_cat = best_model.named_steps['preprocessor'].named_transformers_['cat'] \
    .named_steps['onehot'].get_feature_names_out(categorical)
all_feature_names = np.concatenate([feature_names_num, feature_names_cat])

# Ordina feature per importanza
sorted_idx = importances.argsort()[::-1]

print("\nTop 10 feature importances:")
for idx in sorted_idx[:10]:
    print(f"{all_feature_names[idx]}: {importances[idx]:.4f}")

# 9. (Opzionale) Visualizza matrice di correlazione su train set
corr = X_train[numerical].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .75})
plt.title('Matrice di correlazione (train set)')
plt.show()
