import pandas as pd

TRAIN_PATH = "/morf-data/labels-train-with-underscore.csv"
TEST_PATH = "/morf-data/labels-test-with-underscore.csv"

def build_course_dataset(course_session, label_type):
    """
    Combine the feature dataframe with the label dataframe.
    :param course_session: the name of the course session.
    :param label_type: the type of label to predict.
    :return: a DataFrame with both feature columns and a label column.
    """
    df_labels = pd \
        .concat([pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)]) \
        .query("label_type == @label_type") \
        .set_index("userID") \
        .loc[:, ["label_value"]]

    df_all_data = pd \
        .read_csv("features.csv", index_col = "session_user_id") \
        .merge(df_labels, left_index = True, right_index = True)

    df_all_data.to_csv("all_data.csv", index = False)