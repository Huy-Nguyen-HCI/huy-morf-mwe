import os
from workflow.extraction.utils.sql_utils import extract_coursera_sql_data
from workflow.extraction.utils.feature_utils import combine_features

TEMP_DIR = "/temp-data/"

def extract_data(course_session):
    """
    Extract the student data from various databases into a single feature dataframe.
    :param course_session: the name of the course session.
    """
    
    # create an output directory to store feature files
    output_dir = os.path.join(TEMP_DIR, course_session)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    os.chdir(output_dir)

    # extract features from the MySQL database
    extract_coursera_sql_data(course_session)

    # combine the features into a single file
    combine_features()