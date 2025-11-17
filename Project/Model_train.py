import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, accuracy_score, classification_report, silhouette_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, GridSearchCV
from sklearn.decomposition import PCA as SKPCA
from sklearn.preprocessing import FunctionTransformer
# Resolve paths relative to this script file so the script works regardless of CWD
base_dir = Path(__file__).resolve().parent

student_path = base_dir / "StudentsPerformance.csv"
placement_path = base_dir / "CampusRecruitment.csv"

# Some datasets in this project use a different filename for placement data — try a fallback
placement_fallback = base_dir / "Placement_Data_Full_Class.csv"
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, accuracy_score, classification_report, silhouette_score


# Resolve paths relative to this script file so the script works regardless of CWD
base_dir = Path(__file__).resolve().parent

student_path = base_dir / "StudentsPerformance.csv"
placement_path = base_dir / "CampusRecruitment.csv"
placement_fallback = base_dir / "Placement_Data_Full_Class.csv"

if not student_path.exists():
    raise FileNotFoundError(
        f"StudentsPerformance.csv not found at {student_path}. Make sure the file is in the 'Project' folder or adjust the path.")

if not placement_path.exists():
    if placement_fallback.exists():
        placement_path = placement_fallback
    else:
        raise FileNotFoundError(
            f"Placement dataset not found. Tried: {placement_path} and {placement_fallback}.")

student_df = pd.read_csv(student_path)
placement_df = pd.read_csv(placement_path)

print("Loaded students dataset from:", student_path)
print(student_df.head())
print('\nLoaded placement dataset from:', placement_path)
print(placement_df.head())

# Basic cleaning
student_df.dropna(inplace=True)
# Do not drop rows from placement_df — salary is NaN for 'Not Placed' rows.
# Impute salary with 0 (or median) so we don't lose the 'Not Placed' class.
if 'salary' in placement_df.columns:
    # Add flag for missing salary before imputation
    placement_df.loc[:, 'salary_missing'] = placement_df['salary'].isna().astype(int)
    # Median impute salary (keeps distribution and avoids creating an artificial zero signal)
    median_salary = placement_df['salary'].median()
    placement_df.loc[:, 'salary'] = placement_df['salary'].fillna(median_salary)
    # add a log-transformed salary feature to stabilize scale
    placement_df.loc[:, 'salary_log'] = np.log1p(placement_df['salary'])
student_df.drop_duplicates(inplace=True)
placement_df.drop_duplicates(inplace=True)


################################################################################
# Part 1 — Regression on StudentsPerformance (predict 'math score')
################################################################################
print("\n== Student: regression task (predict 'math score') ==")

if 'math score' not in student_df.columns:
    print("'math score' column not found in students dataset — skipping student regression.")
else:
    # Prepare X and y
    y_student = student_df['math score']
    X_student = student_df.drop(columns=['math score'])

    # Identify categorical and numeric features
    cat_cols_st = X_student.select_dtypes(include=['object']).columns.tolist()
    num_cols_st = X_student.select_dtypes(include=[np.number]).columns.tolist()

    # Preprocessor: impute numeric, scale, and one-hot encode categories (dense output)
    numeric_pipeline_st = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_encoder = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)

    preprocessor_st = ColumnTransformer(
        transformers=[
            ('cat', cat_encoder, cat_cols_st),
            ('num', numeric_pipeline_st, num_cols_st)
        ],
        remainder='drop'
    )

    # Use Ridge instead of plain LinearRegression for numerical stability
    pipeline_st = Pipeline([
        ('pre', preprocessor_st),
        # remove near-constant features after encoding
        ('var', VarianceThreshold(threshold=0.0)),
        ('model', Ridge(alpha=1.0))
    ])

    # Evaluate regression with cross-validation (KFold)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline_st, X_student, y_student, cv=kf, scoring='r2')
    print("Student regression CV R2 mean/std:", cv_scores.mean(), cv_scores.std())

    X_train, X_test, y_train, y_test = train_test_split(X_student, y_student, test_size=0.2, random_state=42)
    try:
        pipeline_st.fit(X_train, y_train)
        y_pred = pipeline_st.predict(X_test)
        print("Student regression R2 Score:", r2_score(y_test, y_pred))
    except Exception as e:
        print("Student regression failed:", e)


################################################################################
# Part 2 — Classification on Placement dataset (predict 'status')
################################################################################
print("\n== Placement: classification task (predict 'status') ==")

if 'status' not in placement_df.columns:
    print("'status' column not found in placement dataset — skipping placement classification.")
else:
    # Prepare data
    X_placement = placement_df.copy()
    # Drop obvious non-feature columns; keep 'salary' as a numeric feature
    for c in ['sl_no']:
        if c in X_placement.columns:
            X_placement.drop(columns=[c], inplace=True)

    y_placement = X_placement.pop('status').map({'Placed': 1, 'Not Placed': 0})

    # Identify categorical and numeric features
    cat_cols_pl = X_placement.select_dtypes(include=['object']).columns.tolist()
    num_cols_pl = X_placement.select_dtypes(include=[np.number]).columns.tolist()

    numeric_pipeline_pl = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_encoder_pl = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)

    # ensure salary_log is included as numeric if present
    if 'salary_log' in X_placement.columns and 'salary' in num_cols_pl:
        # replace salary with salary_log in numeric cols list
        num_cols_pl = [c for c in num_cols_pl if c != 'salary'] + ['salary_log']

    preprocessor_pl = ColumnTransformer(
        transformers=[
            ('cat', cat_encoder_pl, cat_cols_pl),
            ('num', numeric_pipeline_pl, num_cols_pl)
        ],
        remainder='drop'
    )

    # Use stratified split for classification
    X_train, X_test, y_train, y_test = train_test_split(
        X_placement, y_placement, test_size=0.2, random_state=42, stratify=y_placement)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel='linear', probability=True)
    }

    # Check number of classes in the target before training
    unique_classes = pd.Series(y_placement).dropna().unique()
    if len(unique_classes) <= 1:
        print(f"Placement target 'status' contains only one class: {unique_classes}. Skipping classification models.")
    else:
        # Cross-validate classifiers with stratified folds
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # We'll tune LogisticRegression using GridSearch with PCA to reduce dimensionality
        log_pipe = Pipeline([
            ('pre', preprocessor_pl),
            ('var', VarianceThreshold(threshold=0.0)),
            ('pca', SKPCA(n_components=0.95, svd_solver='full')),
            ('model', LogisticRegression(max_iter=2000))
        ])

        param_grid = {
            'model__C': [0.01, 0.1, 1, 10],
            'model__penalty': ['l2']
        }

        try:
            gs = GridSearchCV(log_pipe, param_grid, cv=skf, scoring='accuracy', n_jobs=-1)
            gs.fit(X_placement, y_placement)
            print(f"\nLogistic Regression GridSearch best CV accuracy: {gs.best_score_:.3f} with params {gs.best_params_}")
            # Evaluate best estimator on held-out test set
            best = gs.best_estimator_
            best.fit(X_train, y_train)
            preds = best.predict(X_test)
            print(f"Logistic Regression Test Accuracy: {accuracy_score(y_test, preds):.3f}")
            print(classification_report(y_test, preds, zero_division=0))
        except Exception as e:
            print("Logistic GridSearch failed:", e)

        # For KNN and SVM keep simpler pipeline (no PCA gridsearch) but use variance threshold
        for name, model in [('KNN', KNeighborsClassifier(n_neighbors=5)), ('SVM', SVC(kernel='linear', probability=True))]:
            pipe = Pipeline([('pre', preprocessor_pl), ('var', VarianceThreshold(threshold=0.0)), ('model', model)])
            try:
                cv_scores = cross_val_score(pipe, X_placement, y_placement, cv=skf, scoring='accuracy')
                print(f"{name} CV Accuracy mean/std: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)
                print(f"{name} Test Accuracy: {accuracy_score(y_test, preds):.3f}")
                print(classification_report(y_test, preds, zero_division=0))
            except Exception as e:
                print(f"Model {name} failed: {e}")

    # Clustering (use preprocessed placement features)
    try:
        preprocessed = preprocessor_pl.fit_transform(X_placement)
        if not np.all(np.isfinite(preprocessed)):
            raise ValueError('Non-finite values present in preprocessed placement data')
        # apply variance threshold before clustering
        preprocessed = VarianceThreshold(threshold=0.0).fit_transform(preprocessed)
        # KMeans expects numeric array
        kmeans = KMeans(n_clusters=3, random_state=42)
        cluster_labels = kmeans.fit_predict(preprocessed)
        placement_df['Cluster'] = cluster_labels
        if len(set(cluster_labels)) > 1:
            print("Silhouette Score:", silhouette_score(preprocessed, cluster_labels))
        else:
            print("Silhouette score not defined (only one cluster predicted)")
    except Exception as e:
        print("Clustering skipped due to error:", e)
