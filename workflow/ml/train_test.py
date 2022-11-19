import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.metrics import f1_score

SEED = 1

candidate_models = [
    (LogisticRegression(random_state = SEED), {"C" : [1, 10, 100], "max_iter" : [100, 1000]}),
    (RandomForestClassifier(random_state = SEED), {"n_estimators" : [10, 50, 100], "max_depth" : [10, 20, 30]}),
    (KNeighborsClassifier(), {"n_neighbors" : [5, 10, 15], "weights" : ["uniform", "distance"]})
]

cv_inner = KFold(n_splits = 10, shuffle = True, random_state = SEED)


def select_best_model(X_train, X_val, y_train, y_val):
    """
    For each base estimator, perform grid-search CV to find the hyperparameters that work best on the train set.
    Select the base estimator and associated hyperparameters that work best on the validation set.
    :param X_train: the training feature dataset
    :param X_val: the validation feature dataset
    :param y_train: the training labels
    :param y_val: the validation labels
    """
    best_model = None
    best_val_score = 0
    for (base_estimator, param_grid) in candidate_models:
        search = GridSearchCV(base_estimator, param_grid, cv = cv_inner, scoring = 'f1')
        result = search.fit(X_train, y_train)
        best_estimator = result.best_estimator_
        y_val_pred = best_estimator.predict(X_val)
        val_score = f1_score(y_val, y_val_pred)
        if val_score > best_val_score:
            best_model = best_estimator
    return best_model


def build_predictive_model(course_session, label_type):
    """
    Build a predictive model through model selection and hyperparameter tuning.
    Evaluate the model on a test dataset and save the prediction output to csv.
    :param course_session: the name of the course session.
    :param label_type: the type of label to predict.
    :return:
    """
    df_features = pd.read_csv("all_data.csv")
    X = df_features.drop(columns = "label_value").values
    y = df_features["label_value"].values
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, stratify = y, random_state = SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, random_state = SEED
    )

    best_model = select_best_model(X_train, X_val, y_train, y_val)
    best_model.fit(X_train_val, y_train_val)
    y_pred = best_model.predict(X_test)
    y_score = best_model.predict_proba(X_test)
    output = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred, 'y_score': y_score[:, 1]})
    output.to_csv('/output/output.csv', index = False)
