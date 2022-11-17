import logging, requests, os
import pandas as pd


TEMP_DIR = "/temp-data/"
URI = 'http://course_database_proxy/sql'

def execute_mysql_query(query, db_name):
	data = {"query": query, "db_name": db_name, "course_platform": "coursera"}
	return requests.get(URI, data = data).json()


def extract_coursera_sql_data(course_session):
	logging.basicConfig(filename='/output/logs.log', level=logging.DEBUG)
	extract_quiz_data(course_session)


def extract_quiz_data(course_name):
	query = """SELECT id, Course FROM quiz_metadata LIMIT 10"""
	quiz_info = execute_mysql_query(query, course)
	# df_quiz = pd.DataFrame(list(quiz_info[2]), columns=forum_comments[:2])