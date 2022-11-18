import os
from workflow.extraction.utils.sql_utils import extract_coursera_sql_data
from workflow.extraction.utils.feature_utils import extract_features

def extract_data(course_session):
    extract_coursera_sql_data(course_session)
    extract_features(course_session)