import os
from workflow.extraction.utils.sql_utils import extract_coursera_sql_data
# from workflow.extraction.utils.feature_utils import main as extract_features

def extract_sessions(course_name):
    # loop through all the children session directory
    data_dir = '/morf-data/'
    for entry in os.listdir(data_dir):
        # entry will be like 'microecon_001'
        path = os.path.join(data_dir, entry)
        if entry.startswith(course_name) and os.path.isdir(path):
            extract_coursera_sql_data(entry)
            return
            # extract_features(course_name, entry)