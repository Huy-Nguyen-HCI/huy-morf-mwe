from workflow.extraction.extractors import extract_sessions
# from workflow.extraction.utils.dataset_utils import build_course_dataset
# from workflow.ml.train_test import train_test_course
import pandas as pd
import numpy as np

if __name__ == "__main__":
    course_name = 'microecon'
    label_type = 'dropout'
    extract_sessions(course_name)
    
    df_out = pd.DataFrame({
        "y_true" : np.random.choice([0, 1], size = 20000),
        "y_pred" : np.random.choice([0, 1], size = 20000),
        "y_score" : np.random.random(20000)
    })
    df_out.to_csv("/output/output.csv", index = False)
