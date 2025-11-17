# train_models.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, accuracy_score, classification_report
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
import joblib

base = Path(__file__).resolve().parent

# helper to find dataset paths
def find_file(candidates):
    for p in candidates:
        p = Path(p)
        if not p.is_absolute():
            p = base / p
        if p.exists():
            return p
    return None

# Allow optional CLI args to explicitly point to dataset files. If provided,
# they will be tried first.
import argparse
parser = argparse.ArgumentParser(description='Train student and placement models')
parser.add_argument('--students', help='Path to students-performance CSV', default=None)
parser.add_argument('--campus', help='Path to campus placement CSV', default=None)
args = parser.parse_args()

student_candidates = [
    'datasets/students-performance-in-exams.csv',
    'StudentsPerformance.csv',
    'Project/StudentsPerformance.csv',
    base / 'datasets' / 'students-performance-in-exams.csv'
]
campus_candidates = [
    'datasets/factors-affecting-campus-placement.csv',
    'Placement_Data_Full_Class.csv',
    'Project/Placement_Data_Full_Class.csv',
    base / 'datasets' / 'factors-affecting-campus-placement.csv'
]

if args.students:
    student_candidates.insert(0, args.students)
if args.campus:
    campus_candidates.insert(0, args.campus)

student_path = find_file(student_candidates)
campus_path = find_file(campus_candidates)
if student_path is None or campus_path is None:
    raise FileNotFoundError(f"Could not find dataset files. Tried: {student_candidates} and {campus_candidates}")

student_df = pd.read_csv(student_path)
campus_df = pd.read_csv(campus_path)

print("Loaded student dataset:", student_path, "shape=", student_df.shape)
print("Loaded campus dataset:", campus_path, "shape=", campus_df.shape)

# Basic cleaning
student_df.drop_duplicates(inplace=True)
campus_df.drop_duplicates(inplace=True)

# numeric-only fillna
student_df.fillna(student_df.mean(numeric_only=True), inplace=True)
campus_df.fillna(campus_df.mean(numeric_only=True), inplace=True)

# --- Strategy change: train on the two datasets separately
# Rationale: merging by non-unique keys (gender) causes combinatorial explosion
# and numeric instability. We'll train the academic regressor on the students
# dataset and the placement classifier on the campus dataset independently.

print('Training student academic regressor from students dataset')

# Students dataset preprocessing and regression
students = student_df.copy()
students.reset_index(drop=True, inplace=True)
students.drop_duplicates(inplace=True)
students.fillna(students.mean(numeric_only=True), inplace=True)

# Targets: predict 'math score' from other cols
if 'math score' in students.columns:
    target = 'math score'
    # categorical columns to encode
    cat_cols = [c for c in students.select_dtypes(include=['object']).columns if c != target]
    num_cols = [c for c in students.select_dtypes(include=[np.number]).columns if c != target]

    # Clip extreme values (1st-99th percentile) and replace inf/nan with median per column
    for c in num_cols:
        # avoid chained-assignment warnings by assigning back to the column explicitly
        col = students[c].replace([np.inf, -np.inf], np.nan)
        q_low = col.quantile(0.01)
        q_high = col.quantile(0.99)
        col = col.clip(lower=q_low, upper=q_high)
        med = col.median()
        col = col.fillna(med)
        students.loc[:, c] = col

    preproc_students = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ('num', RobustScaler(), num_cols)
    ], remainder='drop')

    # Transformer to sanitize non-finite values after preprocessing
    class NumericSanitizer(BaseEstimator, TransformerMixin):
        def __init__(self):
            self.medians_ = None
        def fit(self, X, y=None):
            # compute medians per column (X is numpy array)
            X = np.asarray(X)
            # if 1D, reshape
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            # compute medians ignoring nan/inf
            self.medians_ = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
            # replace nan medians with 0
            self.medians_ = np.where(np.isfinite(self.medians_), self.medians_, 0.0)
            return self
        def transform(self, X):
            X = np.asarray(X).astype(float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            mask = ~np.isfinite(X)
            if np.any(mask):
                # broadcast medians
                X[mask] = np.take(self.medians_, np.where(mask)[1])
            return X

    student_pipeline = Pipeline([
        ('pre', preproc_students),
        ('san', NumericSanitizer()),
        ('var', VarianceThreshold(threshold=1e-3)),
        ('reg', Ridge(alpha=1.0))
    ])

    Xs = students.drop(columns=[target])
    ys = students[target].astype(float)
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(Xs, ys, test_size=0.2, random_state=42)

    student_pipeline.fit(Xs_train, ys_train)
    ys_pred = student_pipeline.predict(Xs_test)
    print('Student regression R2:', r2_score(ys_test, ys_pred))
    # Save the pipeline (includes preprocessor and regressor)
    joblib.dump(student_pipeline, base / 'academic_regressor.pkl')
else:
    print('No math score column found in students dataset — skipping student regression')

print('\nTraining placement classifier from campus dataset')

# Campus placement preprocessing and classification
campus = campus_df.copy()
campus.reset_index(drop=True, inplace=True)
campus.drop_duplicates(inplace=True)
campus.fillna(campus.mean(numeric_only=True), inplace=True)

# Exclude salary and salary_missing to avoid leakage
if 'salary' in campus.columns:
    campus['salary'] = campus['salary'].fillna(campus['salary'].median())

if 'status' in campus.columns:
    target_cls = 'status'

    cat_cols_c = [c for c in campus.select_dtypes(include=['object']).columns if c != target_cls]
    num_cols_c = [c for c in campus.select_dtypes(include=[np.number]).columns if c not in ['salary']]

    # Clip and sanitize numeric columns
    for c in num_cols_c:
        col = campus[c].replace([np.inf, -np.inf], np.nan)
        q_low = col.quantile(0.01)
        q_high = col.quantile(0.99)
        col = col.clip(lower=q_low, upper=q_high)
        med = col.median()
        col = col.fillna(med)
        campus.loc[:, c] = col

    preproc_campus = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols_c),
        ('num', RobustScaler(), num_cols_c)
    ], remainder='drop')

    cls_pipeline = Pipeline([
        ('pre', preproc_campus),
        ('san', NumericSanitizer()),
        ('var', VarianceThreshold(threshold=1e-3)),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', solver='saga'))
    ])

    Xc = campus.drop(columns=[target_cls, 'salary'], errors='ignore')
    yc = campus[target_cls].astype(str)
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)

    cls_pipeline.fit(Xc_train, yc_train)
    yc_pred = cls_pipeline.predict(Xc_test)
    print('Placement classifier accuracy:', accuracy_score(yc_test, yc_pred))
    print(classification_report(yc_test, yc_pred, zero_division=0))
    joblib.dump(cls_pipeline, base / 'placement_log_model.pkl')
else:
    print('No status column found in campus dataset — skipping placement classification')

## Post-training: optional clustering and saving of preprocessors
try:
    if 'student_pipeline' in locals():
        # Save student preprocessor pipeline
        joblib.dump(student_pipeline, base / 'academic_regressor_pipeline.pkl')
    if 'cls_pipeline' in locals():
        joblib.dump(cls_pipeline, base / 'placement_pipeline.pkl')

    # Clustering on students using the preprocessed features if available
    if 'student_pipeline' in locals():
        Xs_all = students.drop(columns=['math score']) if 'math score' in students.columns else students
        pre = student_pipeline.named_steps['pre']
        Xs_trans = pre.transform(Xs_all)
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(Xs_trans)
        students['cluster'] = clusters
        joblib.dump(kmeans, base / 'student_cluster_model.pkl')
        print('KMeans cluster centers shape:', kmeans.cluster_centers_.shape)
except Exception as e:
    print('Post-training save/clustering failed:', e)

print('All done — models saved to', base)
