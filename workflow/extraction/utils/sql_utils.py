# Copyright (c) 2018 The Regents of the University of Michigan
# and the University of Pennsylvania
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging, requests, os
import pandas as pd

DB_URI = 'http://course_database_proxy/sql'

def execute_mysql_query(query, db_name):
	"""
	Send a MySQL query to the specified course database.
	:param query: the query string.
	:param db_name: the database name, which is also the course session name, e.g., accounting_001
	:return: the SQL output in JSON format.
	"""
	data = {"query": query, "db_name": db_name, "course_platform": "coursera"}
	return requests.get(DB_URI, data = data).json()


def extract_quiz_submissions(course_session):
	"""
	Extract the quiz id, student id and quiz score from the quiz submission database. Save the result to csv.
	:param course_session: the name of the course session.
	:return:
	"""
	query = """SELECT item_id, session_user_id, raw_score FROM quiz_submission_metadata"""
	response = execute_mysql_query(query, course_session)
	df = pd.DataFrame(list(response[3]), columns = response[:3])
	df.to_csv("quiz_submissions.csv", index = False)


def extract_lecture_viewing(course_session):
	"""
	Extract the lecture id, student id and number of watching times from the lecture database. Save the result to csv.
	:param course_session: the name of the course session.
	:return:
	"""
	query = """SELECT item_id, session_user_id, submission_number FROM lecture_submission_metadata"""
	response = execute_mysql_query(query, course_session)
	df = pd.DataFrame(list(response[3]), columns = response[:3])
	df.to_csv("lecture_viewings.csv", index = False)


def extract_course_grades(course_session):
	"""
	Extract the student id, normal grade and distinction grade from the course grade database. Save the result to csv.
	:param course_session: the name of the course session.
	:return:
	"""
	query = """SELECT session_user_id, normal_grade, distinction_grade FROM course_grades"""
	response = execute_mysql_query(query, course_session)
	df = pd.DataFrame(list(response[3]), columns = response[:3])
	df.to_csv("course_grades.csv", index = False)


def extract_coursera_sql_data(course_session):
	"""
	Extract data about quiz submission, lecture viewing and course grade. Save the results to csv files.
	:param course_session: the name of the course session.
	:return:
	"""
	logging.basicConfig(filename='/output/logs.log', level=logging.DEBUG)
	extract_quiz_submissions(course_session)
	extract_lecture_viewing(course_session)
	extract_course_grades(course_session)
