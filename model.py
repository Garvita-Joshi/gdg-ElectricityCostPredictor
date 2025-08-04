import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor

# Loading the dataset for training
df = pd.read_csv("electricity_cost_dataset.csv")

# Define target and features
target = "electricity cost"
# excluding the target variable from features
X = df.drop(columns=[target])
y = df[target]

# Seprating the categorical and numerical features
# it was made clear that 'structure type' is a categorical feature vis the df.head() from the analysis file
categorical_features = ["structure type"]
numerical_features = [col for col in X.columns if col not in categorical_features]

# Preprocessing: Scale numerics + OneHotEncode categoricals
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(drop="first"), categorical_features)
])

# Choosing Gradient Boosting Regressor to train the model
# I had tried other traing methods like RandomForestRegressor, Linear regression
# and Descision tree but GradientBoostingRegressor gave the best results
model = GradientBoostingRegressor(random_state=42)

# Build pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning
param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__learning_rate": [0.05, 0.1, 0.2],
    "model__max_depth": [3, 5, 7]
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=10,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    random_state=42
)

# training the model for each hyperparameter 
search.fit(X_train, y_train)

# Geting the best model from the choosen hyperparamters 
best_model = search.best_estimator_

# Evaluating the model performance 
preds = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

# Displaying the evaluation done 
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.3f}")

# Saving the best model
joblib.dump(best_model, "model.pkl")
print("Model saved as model.pkl")
