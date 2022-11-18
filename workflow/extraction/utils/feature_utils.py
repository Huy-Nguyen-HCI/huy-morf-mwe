import pandas as pd

def extract_features(course_session):
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