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

import pandas as pd

def combine_features():
	"""
	Merge several csv files that contain student features into a single feature dataframe. Save the output to csv.
	:return:
	"""
	df_quiz_submissions = pd \
		.read_csv("quiz_submissions.csv") \
		.pivot_table(index = "session_user_id", columns = "item_id", values = "raw_score", fill_value = 0)

	df_lecture_viewings = pd \
		.read_csv("lecture_viewings.csv") \
		.pivot_table(index = "session_user_id", columns = "item_id", values = "submission_number", fill_value = 0)

	df_course_grades = pd \
		.read_csv("course_grades.csv") \
		.set_index("session_user_id")

	df_features = pd.concat(
		[df_quiz_submissions, df_lecture_viewings, df_course_grades],
		axis = 1, join = "inner"
	)

	df_features.to_csv("features.csv")