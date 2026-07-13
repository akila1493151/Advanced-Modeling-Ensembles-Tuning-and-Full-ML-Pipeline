import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# =========================================================================
# SETUP: Synthetic Data Generation (Simulating Data State from Part 2)
# =========================================================================
print("=== Setup: Generating Synthetic Classification Data ===")
X_raw, y_clf = make_classification(
    n_samples=1000, 
    n_features=12, 
    n_informative=8, 
    n_redundant=2, 
    n_classes=2, 
    random_state=42
)

# Convert to DataFrame for naming consistency and feature ablation tracking
feature_names = [f"Feature_{i}" for i in range(X_raw.shape[1])]
X_df = pd.DataFrame(X_raw, columns=feature_names)

# Inject random missing values into 5% of rows to test the Pipeline Imputer
np.random.seed(42)
mask = np.random.rand(*X_df.shape) < 0.05
X_df[mask] = np.nan

# Split into Train and Test Sets
X_train, X_test, y_clf_train, y_clf_test = train_test_split(
    X_df, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# Create Scaled Variants for raw/un-pipelined model architectures (Tasks 1-4)
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train))
X_test_scaled = scaler.transform(imputer.transform(X_test))

print(f"Train set shape: {X_train.shape} | Test set shape: {X_test.shape}\n")


# =========================================================================
# TASK 1: Decision Tree Baseline (Unconstrained)
# =========================================================================
print("=== Task 1: Decision Tree Baseline (Unconstrained) ===")
dt_unconstrained = DecisionTreeClassifier(default_max_depth=None, random_state=42)
dt_unconstrained.fit(X_train_scaled, y_clf_train)

acc_train_unconstrained = accuracy_score(y_clf_train, dt_unconstrained.predict(X_train_scaled))
acc_test_unconstrained = accuracy_score(y_clf_test, dt_unconstrained.predict(X_test_scaled))

print(f"Training Accuracy : {acc_train_unconstrained * 100:.2f}%")
print(f"Test Accuracy     : {acc_test_unconstrained * 100:.2f}%\n")


# =========================================================================
# TASK 2: Controlled Decision Tree
# =========================================================================
print("=== Task 2: Controlled Decision Tree ===")
dt_controlled = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42)
dt_controlled.fit(X_train_scaled, y_clf_train)

acc_train_controlled = accuracy_score(y_clf_train, dt_controlled.predict(X_train_scaled))
acc_test_controlled = accuracy_score(y_clf_test, dt_controlled.predict(X_test_scaled))

print(f"Controlled Training Accuracy : {acc_train_controlled * 100:.2f}%")
print(f"Controlled Test Accuracy     : {acc_test_controlled * 100:.2f}%\n")


# =========================================================================
# TASK 3: Gini vs Entropy Comparison
# =========================================================================
print("=== Task 3: Gini vs Entropy Comparison ===")
dt_gini = DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=42)
dt_entropy = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=42)

dt_gini.fit(X_train_scaled, y_clf_train)
dt_entropy.fit(X_train_scaled, y_clf_train)

print(f"Gini Criterion Test Accuracy    : {accuracy_score(y_clf_test, dt_gini.predict(X_test_scaled)) * 100:.2f}%")
print(f"Entropy Criterion Test Accuracy : {accuracy_score(y_clf_test, dt_entropy.predict(X_test_scaled)) * 100:.2f}%\n")


# =========================================================================
# TASK 4: Random Forest & Feature Importances
# =========================================================================
print("=== Task 4: Random Forest Classifier ===")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train_scaled, y_clf_train)

rf_train_acc = accuracy_score(y_clf_train, rf_model.predict(X_train_scaled))
rf_test_acc = accuracy_score(y_clf_test, rf_model.predict(X_test_scaled))
rf_test_auc = roc_auc_score(y_clf_test, rf_model.predict_proba(X_test_scaled)[:, 1])

print(f"RF Training Accuracy : {rf_train_acc * 100:.2f}%")
print(f"RF Test Accuracy     : {rf_test_acc * 100:.2f}%")
print(f"RF Test ROC-AUC      : {rf_test_auc:.4f}\n")

# Top 5 Features Evaluation
importances = rf_model.feature_importances_
fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

print("Top 5 Features by Gini Importance:")
print(fi_df.head(5).to_string(index=False))
print("\n")


# =========================================================================
# TASK 4a: Gradient Boosting
# =========================================================================
print("=== Task 4a: Gradient Boosting Classifier ===")
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train_scaled, y_clf_train)

gb_train_acc = accuracy_score(y_clf_train, gb_model.predict(X_train_scaled))
gb_test_acc = accuracy_score(y_clf_test, gb_model.predict(X_test_scaled))
gb_test_auc = roc_auc_score(y_clf_test, gb_model.predict_proba(X_test_scaled)[:, 1])

print(f"GB Training Accuracy : {gb_train_acc * 100:.2f}%")
print(f"GB Test Accuracy     : {gb_test_acc * 100:.2f}%")
print(f"GB Test ROC-AUC      : {gb_test_auc:.4f}\n")


# =========================================================================
# TASK 4b: Feature Ablation Study
# =========================================================================
print("=== Task 4b: Feature Ablation Study ===")
lowest_5_features = fi_df.tail(5)['Feature'].tolist()
print(f"5 Features with lowest importance scores: {lowest_5_features}")

# Filter out features by indexing column locations
low_imp_indices = [feature_names.index(f) for f in lowest_5_features]
X_train_reduced = np.delete(X_train_scaled, low_imp_indices, axis=1)
X_test_reduced = np.delete(X_test_scaled, low_imp_indices, axis=1)

# Retrain identical Random Forest architecture on reduced dataset
rf_reduced = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_reduced.fit(X_train_reduced, y_clf_train)
rf_reduced_auc = roc_auc_score(y_clf_test, rf_reduced.predict_proba(X_test_reduced)[:, 1])

print(f"Full Feature Model ROC-AUC   : {rf_test_auc:.4f}")
print(f"Reduced Feature Model ROC-AUC: {rf_reduced_auc:.4f}\n")


# =========================================================================
# TASK 5: Cross-Validated Comparison Matrix
# =========================================================================
print("=== Task 5: Cross-Validated Comparison Matrix ===")
lr_baseline = LogisticRegression(random_state=42)

models_to_cv = {
    'Logistic Regression': lr_baseline,
    'Controlled Decision Tree': dt_controlled,
    'Random Forest': rf_model,
    'Gradient Boosting': gb_model
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, model in models_to_cv.items():
    scores = cross_val_score(model, X_train_scaled, y_clf_train, cv=cv_strategy, scoring='roc_auc', n_jobs=-1)
    cv_results[name] = {
        'Mean AUC': scores.mean(),
        'Std AUC': scores.std()
    }
    print(f"{name:25} | 5-Fold Mean ROC-AUC: {scores.mean():.4f} (±{scores.std():.4f})")
print("\n")


# =========================================================================
# TASK 6: Hyperparameter Tuning with GridSearchCV over Pipeline
# =========================================================================
print("=== Task 6: Hyperparameter Tuning with GridSearchCV ===")

# Construct reproducible Scikit-Learn Pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('randomforestclassifier', RandomForestClassifier(random_state=42))
])

param_grid = {
    'randomforestclassifier__n_estimators': [50, 100, 200],
    'randomforestclassifier__max_depth': [5, 10, None],
    'randomforestclassifier__min_samples_leaf': [1, 5]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv_strategy,
    scoring='roc_auc',
    n_jobs=-1
)

# Fit on Raw Data frames (Pipeline cleanly internalizes imputation and scaling transforms)
grid_search.fit(X_train, y_clf_train)

print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Best Out-Of-Fold Cross-Validation ROC-AUC: {grid_search.best_score_:.4f}\n")

best_pipeline = grid_search.best_estimator__


# =========================================================================
# TASK 7: Manual Learning Curve Evaluation
# =========================================================================
print("=== Task 7: Manual Learning Curve Table ===")
fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
learning_curve_records = []

for f in fractions:
    subset_size = int(f * len(X_train))
    X_train_sub = X_train.iloc[:subset_size]
    y_train_sub = y_clf_train[:subset_size]
    
    # Clone and re-fit optimization pipeline states independently
    cloned_pipeline = joblib.from_pickle(joblib.to_pickle(best_pipeline))
    cloned_pipeline.fit(X_train_sub, y_train_sub)
    
    # Compute Evaluation Diagnostics
    train_auc = roc_auc_score(y_train_sub, cloned_pipeline.predict_proba(X_train_sub)[:, 1])
    test_auc = roc_auc_score(y_clf_test, cloned_pipeline.predict_proba(X_test)[:, 1])
    
    learning_curve_records.append({
        'Training Fraction': f,
        'Training AUC': round(train_auc, 4),
        'Test AUC': round(test_auc, 4)
    })

lc_df = pd.DataFrame(learning_curve_records)
print(lc_df.to_string(index=False))
print("\n")


# =========================================================================
# TASK 8: Serialize Best Model Assets & Mock Verification
# =========================================================================
print("=== Task 8: Model Serialization & Reload Verification ===")
model_filename = 'best_model.pkl'

# Serialize the optimized production-ready pipeline
joblib.dump(best_pipeline, model_filename)
print(f"SUCCESS: Production pipeline saved to disc as '{model_filename}'")

# --- Reload and Predict Verification Code Block ---
loaded_model = joblib.load(model_filename)

# Construct 2 hand-crafted synthetic evaluation samples matching the input schema
mock_data = pd.DataFrame(
    np.random.randn(2, len(feature_names)), 
    columns=feature_names
)
# Intentionally inject an unexpected missing value to demonstrate pipeline robustness
mock_data.iloc[0, 2] = np.nan 

mock_predictions = loaded_model.predict(mock_data)
mock_probabilities = loaded_model.predict_proba(mock_data)[:, 1]

print("\n--- Verification Predictions Execution on Mock Rows ---")
print(f"Generated Vector Categorical Classes Output: {mock_predictions}")
print(f"Generated Vector Class Probability Output  : {mock_probabilities}")
print("SUCCESS: Reload pipeline executed smoothly without internal state discrepancies.")
