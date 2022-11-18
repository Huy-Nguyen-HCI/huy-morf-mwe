import logging, requests, os
import pandas as pd


TEMP_DIR = "/temp-data/"
DB_URI = 'http://course_database_proxy/sql'

def execute_mysql_query(query, db_name):
	data = {"query": query, "db_name": db_name, "course_platform": "coursera"}
	return requests.get(DB_URI, data = data).json()


def extract_quiz_submissions(course_session, output_dir):
	query = """SELECT item_id, session_user_id, raw_score FROM quiz_submission_metadata"""
	response = execute_mysql_query(query, course_session)
	df = pd.DataFrame(list(response[3]), columns = response[:3])
	df.to_csv("quiz_submissions.csv", index = False)


def extract_lecture_viewing(course_session, output_dir):
	query = """SELECT item_id, session_user_id, submission_number FROM lecture_submission_metadata"""
	response = execute_mysql_query(query, course_session)
	df = pd.DataFrame(list(response[3]), columns = response[:3])
	df.to_csv("lecture_viewings.csv", index = False)


def extract_course_grades(course_session, output_dir):
	query = """SELECT session_user_id, normal_grade, distinction_grade FROM course_grades"""
	response = execute_mysql_query(query, course_session)
	df = pd.DataFrame(list(response[3]), columns = response[:3])
	df.to_csv("course_grades.csv", index = False)


def extract_coursera_sql_data(course_session):
	logging.basicConfig(filename='/output/logs.log', level=logging.DEBUG)
	output_dir = os.path.join(TEMP_DIR, course_session)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	os.chdir(output_dir)
	extract_quiz_submissions(course_session, output_dir)
	extract_lecture_viewing(course_session, output_dir)
	extract_course_grades(course_session, output_dir)
