from workflow.extraction.extractors import extract_data
from workflow.extraction.utils.dataset_utils import build_course_dataset
from workflow.ml.train_test import build_predictive_model
import pandas as pd
import numpy as np

def build_dummy_output():
    """
    Create an output.csv with random label values and prediction scores.
    This helps with MORF's final evaluation while you are working on earlier steps in the pipeline.
    :return:
    """
    df_out = pd.DataFrame({
        "y_true" : np.random.choice([0, 1], size = 20000),
        "y_pred" : np.random.choice([0, 1], size = 20000),
        "y_score" : np.random.random(20000)
    })
    df_out.to_csv("/output/output.csv", index = False)

if __name__ == "__main__":
    course_session = 'accounting_001'
    label_type = 'dropout'
    extract_data(course_session)
    build_course_dataset(course_session, label_type)
    build_predictive_model(course_session, label_type)
    
    # build_dummy_output()

