# Advanced-Modeling-Ensembles-Tuning-and-Full-ML-Pipeline
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

# Note: Ensure X_train, X_test, X_train_scaled, X_test_scaled, y_clf_train, y_clf_test are defined.
# If you don't have them yet, mock data for validation:
# X_train = pd.DataFrame(np.random.randn(200, 10), columns=[f'feat_{i}' for i in range(10)])
# X_test = pd.DataFrame(np.random.randn(50, 10), columns=[f'feat_{i}' for i in range(10)])
# X_train_scaled, X_test_scaled = X_train.values, X_test.values
# y_clf_train = np.random.randint(0, 2, 200)
# y_clf_test = np.random.randint(0, 2, 50)

# -------------------------------------------------------------------------
# Task 1: Decision Tree baseline
# -------------------------------------------------------------------------
dt_baseline = DecisionTreeClassifier(random_state=42)
dt_baseline.fit(X_train_scaled, y_clf_train)

dt_b_train_acc = accuracy_score(y_clf_train, dt_baseline.predict(X_train_scaled))
dt_b_test_acc = accuracy_score(y_clf_test, dt_baseline.predict(X_test_scaled))

print("=== Task 1: Decision Tree Baseline ===")
print(f"Train Accuracy: {dt_b_train_acc:.4f} | Test Accuracy: {dt_b_test_acc:.4f}\n")


# -------------------------------------------------------------------------
# Task 2: Controlled Decision Tree
# -------------------------------------------------------------------------
dt_controlled = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42)
dt_controlled.fit(X_train_scaled, y_clf_train)

dt_c_train_acc = accuracy_score(y_clf_train, dt_controlled.predict(X_train_scaled))
dt_c_test_acc = accuracy_score(y_clf_test, dt_controlled.predict(X_test_scaled))

print("=== Task 2: Controlled Decision Tree ===")
print(f"Train Accuracy: {dt_c_train_acc:.4f} | Test Accuracy: {dt_c_test_acc:.4f}\n")


# -------------------------------------------------------------------------
# Task 3: Gini vs Entropy comparison
# -------------------------------------------------------------------------
dt_gini = DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=42)
dt_entropy = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=42)

dt_gini.fit(X_train_scaled, y_clf_train)
dt_entropy.fit(X_train_scaled, y_clf_train)

gini_test_acc = accuracy_score(y_clf_test, dt_gini.predict(X_test_scaled))
entropy_test_acc = accuracy_score(y_clf_test, dt_entropy.predict(X_test_scaled))

print("=== Task 3: Gini vs Entropy ===")
print(f"Gini Test Accuracy: {gini_test_acc:.4f} | Entropy Test Accuracy: {entropy_test_acc:.4f}\n")


# -------------------------------------------------------------------------
# Task 4: Random Forest & Feature Importances
# -------------------------------------------------------------------------
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train_scaled, y_clf_train)

rf_train_acc = accuracy_score(y_clf_train, rf_model.predict(X_train_scaled))
rf_test_acc = accuracy_score(y_clf_test, rf_model.predict(X_test_scaled))
rf_test_auc = roc_auc_score(y_clf_test, rf_model.predict_proba(X_test_scaled)[:, 1])

print("=== Task 4: Random Forest ===")
print(f"Train Acc: {rf_train_acc:.4f} | Test Acc: {rf_test_acc:.4f} | Test AUC: {rf_test_auc:.4f}")

# Extract feature names if working with DataFrames, else fallback to indices
feature_names = X_train.columns if isinstance(X_train, pd.DataFrame) else [f"feature_{i}" for i in range(X_train_scaled.shape[1])]
importances = rf_model.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

print("\nTop 5 Features by Importance:")
print(feat_imp_df.head(5).to_string(index=False))
print("\n")


# -------------------------------------------------------------------------
# Task 4a: Gradient Boosting
# -------------------------------------------------------------------------
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train_scaled, y_clf_train)

gb_train_acc = accuracy_score(y_clf_train, gb_model.predict(X_train_scaled))
gb_test_acc = accuracy_score(y_clf_test, gb_model.predict(X_test_scaled))
gb_test_auc = roc_auc_score(y_clf_test, gb_model.predict_proba(X_test_scaled)[:, 1])

print("=== Task 4a: Gradient Boosting ===")
print(f"Train Acc: {gb_train_acc:.4f} | Test Acc: {gb_test_acc:.4f} | Test AUC: {gb_test_auc:.4f}\n")


# -------------------------------------------------------------------------
# Task 4b: Feature Ablation Study
# -------------------------------------------------------------------------
lowest_5_features = feat_imp_df.tail(5)['Feature'].tolist()
print("=== Task 4b: Feature Ablation Study ===")
print(f"5 Lowest Importance Features to remove: {lowest_5_features}")

if isinstance(X_train, pd.DataFrame):
    X_train_reduced = X_train.drop(columns=lowest_5_features)
    X_test_reduced = X_test.drop(columns=lowest_5_features)
    # Re-apply scaling for evaluation consistency
    scaler_reduced = StandardScaler().fit(X_train_reduced)
    X_train_reduced_scaled = scaler_reduced.transform(X_train_reduced)
    X_test_reduced_scaled = scaler_reduced.transform(X_test_reduced)
else:
    # If using numpy arrays, map tail indices
    lowest_indices = feat_imp_df.tail(5).index.tolist()
    X_train_reduced_scaled = np.delete(X_train_scaled, lowest_indices, axis=1)
    X_test_reduced_scaled = np.delete(X_test_scaled, lowest_indices, axis=1)

rf_reduced = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_reduced.fit(X_train_reduced_scaled, y_clf_train)
rf_reduced_auc = roc_auc_score(y_clf_test, rf_reduced.predict_proba(X_test_reduced_scaled)[:, 1])

print(f"Full Model Test AUC (All Features): {rf_test_auc:.4f}")
print(f"Reduced Model Test AUC (5 Features Removed): {rf_reduced_auc:.4f}\n")


# -------------------------------------------------------------------------
# Task 5: Cross-validated Comparison
# -------------------------------------------------------------------------
# Assuming a mock / standard Logistic Regression from Part 2 setup
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(max_iter=1000, random_state=42)

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models_to_cv = {
    'Logistic Regression': lr_model,
    'Controlled Decision Tree': dt_controlled,
    'Random Forest': rf_model,
    'Gradient Boosting': gb_model
}

cv_results = {}
print("=== Task 5: 5-Fold Cross-Validation (ROC-AUC) ===")
for name, model in models_to_cv.items():
    scores = cross_val_score(model, X_train_scaled, y_clf_train, cv=cv_strategy, scoring='roc_auc', n_jobs=-1)
    cv_results[name] = (scores.mean(), scores.std())
    print(f"{name} -> Mean AUC: {scores.mean():.4f} | Std AUC: {scores.std():.4f}")
print("\n")


# -------------------------------------------------------------------------
# Task 6: Hyperparameter Tuning with GridSearchCV
# -------------------------------------------------------------------------
param_grid = {
    'randomforestclassifier__n_estimators': [50, 100, 200],
    'randomforestclassifier__max_depth': [5, 10, None],
    'randomforestclassifier__min_samples_leaf': [1, 5]
}

# Build Pipeline (uses unscaled raw features X_train)
pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
    RandomForestClassifier(random_state=42)
)

grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=cv_strategy, 
    scoring='roc_auc', 
    n_jobs=-1
)
grid_search.fit(X_train, y_clf_train)

print("=== Task 6: GridSearchCV Results ===")
print(f"Best Params: {grid_search.best_params_}")
print(f"Best CV Score (AUC): {grid_search.best_score_:.4f}\n")

best_pipeline = grid_search.best_estimator_


# -------------------------------------------------------------------------
# Task 7: Manual Learning Curve
# -------------------------------------------------------------------------
print("=== Task 7: Manual Learning Curve ===")
fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
learning_curve_data = []

n_samples = len(X_train)

for f in fractions:
    subset_size = int(f * n_samples)
    
    # Slice subsets from the raw, unscaled training frames
    if isinstance(X_train, pd.DataFrame):
        X_sub = X_train.iloc[:subset_size]
        y_sub = y_clf_train[:subset_size] if isinstance(y_clf_train, np.ndarray) else y_clf_train.iloc[:subset_size]
    else:
        X_sub = X_train[:subset_size]
        y_sub = y_clf_train[:subset_size]
        
    # Fit pipeline
    best_pipeline.fit(X_sub, y_sub)
    
    # Training AUC on the current subset
    train_preds = best_pipeline.predict_proba(X_sub)[:, 1]
    train_auc = roc_auc_score(y_sub, train_preds)
    
    # Test AUC on fixed full test set (we use X_test raw because pipeline handles scaling)
    test_preds = best_pipeline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_clf_test, test_preds)
    
    learning_curve_data.append({
        'Training fraction': f,
        'Training AUC': round(train_auc, 4),
        'Test AUC': round(test_auc, 4)
    })

lc_df = pd.DataFrame(learning_curve_data)
print(lc_df.to_string(index=False))
print("\n")


# -------------------------------------------------------------------------
# Task 8: Serialize the Best Model & Verification
# -------------------------------------------------------------------------
# Save the model
joblib.dump(best_pipeline, 'best_model.pkl')
print("Model serialized successfully as 'best_model.pkl'.\n")

# Verification Block (At least 5 lines of code)
print("=== Task 8: Reloading & Verifying Model ===")
loaded_pipeline = joblib.load('best_model.pkl')
# Generate two hand-crafted rows with matching dimensions
mock_features = np.random.randn(2, X_train.shape[1]) 
hand_crafted_batch = pd.DataFrame(mock_features, columns=feature_names) if isinstance(X_train, pd.DataFrame) else mock_features
predictions = loaded_pipeline.predict(hand_crafted_batch)
print(f"Predictions for hand-crafted items: {predictions}")
print("Verification block completed successfully with zero errors.")
