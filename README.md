# End-to-End Machine Learning Classification Pipeline

This repository contains an end-to-end Python pipeline demonstrating the progression from a baseline decision tree model to highly optimized, serialized ensemble architectures using `scikit-learn`. The pipeline simulates a realistic workflow by incorporating synthetic data generation, missing value handling, feature ablation, cross-validation, hyperparameter tuning, and robust model serialization.

---

## 📌 Project Overview & Architecture

The script processes data sequentially across 8 structural tasks designed to compare core machine learning methodologies and validate production readiness:

1. **Synthetic Data & Preprocessing Pipeline:** Generates a classification dataset, introduces 5% missingness, and configures `SimpleImputer` and `StandardScaler`.
2. **Decision Tree Baselines:** Compares an unconstrained tree against a regularized (controlled) tree to isolate overfitting behaviors.
3. **Criterion Analysis:** Benchmarks structural variations using `Gini` versus `Entropy` impurity metrics.
4. **Ensemble Modeling & Feature Engineering:** Implements `Random Forest` and `Gradient Boosting` classifiers alongside a **Feature Ablation Study** that evaluates dropping the lowest 5 features.
5. **Cross-Validated Benchmark:** Computes robust `StratifiedKFold` (5-Fold) mean ROC-AUC metrics across all candidate models (including a Logistic Regression baseline).
6. **Hyperparameter Tuning via Pipelines:** Implements an encapsulated, leakage-free `GridSearchCV` over a Scikit-Learn `Pipeline`.
7. **Diagnostic Learning Curves:** Computes training and test metrics incrementally across training set fractions to diagnose bias-variance trade-offs.
8. **Serialization & Production Verification:** Dumps the finalized pipeline configuration to disk and confirms missing-value resilience using simulated mock inferences.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have a Python environment installed (Python 3.8+ recommended). Install the required library ecosystem via `pip`:

```bash
pip install numpy pandas scikit-learn joblib
