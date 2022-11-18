import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

TRAIN_PATH = "/morf-data/labels-train-with-underscore.csv"
TEST_PATH = "/morf-data/labels-test-with-underscore.csv"

def build_dataset_for_prediction(course_session, label_type):
    df_labels = pd \
        .concat([pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)]) \
        .query("label_type == @label_type") \
        .set_index("userID") \
        .loc[:, ["label_value"]]

    return pd \
        .read_csv("features.csv", index_col = "session_user_id") \
        .merge(df_labels, left_index = True, right_index = True)


def build_predictive_model(course_session, label_type):
    df_features = build_dataset_for_prediction(course_session, label_type)
    X = df_features.drop(columns = "label_value").values
    y = df_features["label_value"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify = y, random_state = 1
    )
    clf = LogisticRegression(random_state = 1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)
    output = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred, 'y_score': y_score[:, 1]})
    output.to_csv('/output/output.csv', index = False)
