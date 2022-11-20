# MORF-MWE-WALKTHROUGH

This is a minimum working example walkthrough for the MORF API v0.1.0. We will build a model to predict student dropout in an Accounting course, based on data about their quiz grades, course grades and lecture viewing activities. The best model will be identified through hyperparameter tuning and model selection.


Table of Contents
=================
* [MORF-MWE-WALKTHROUGH](#morf-mwe-walkthrough)
	* [Dockerfile](#dockerfile)
	* [Minimum Working Example - <strong>mwe.py</strong>](#minimum-working-example---mwepy)
		* [Extracting the Data - <strong>extract_data</strong>](#extracting-the-data---extract_data)
		* [Extracting Data from Coursera <strong>extract_coursera_sql_data</strong>](#extracting-data-from-coursera-extract_coursera_sql_data)
		* [Combining Features <strong>combine_features</strong>](#combining-features-combine_features)
	  * [<strong>build_course_dataset</strong>](#build_course_dataset)
	  * [<strong>build_predictive_model</strong>](#build_predictive_model)
   * [After job execution](#after-job-execution)

## Dockerfile
```Dockerfile
FROM ubuntu:20.04

# install Python 
RUN apt update
RUN apt install -y python3-pip
RUN apt update
RUN apt install -y iputils-ping
RUN apt install -y net-tools
RUN apt install -y curl
RUN apt install -y wget
RUN apt install -y nano
RUN pip install requests==2.28.1 pandas==1.5.1 numpy==1.23.4 scikit-learn==1.1.3

# add scripts
ADD mwe.py mwe.py
ADD workflow workflow

# define entrypoint
ENTRYPOINT ["python3", "mwe.py"]
```

The Dockerfile specifies the base Docker image as Ubunty 20.04, then installs `python3-pip` and the necessary Python packages for data science and machine learning. To support replicability, we will also specify the version numbers of all Python packages, because different package versions may yield different results. The version numbers are based on the latest version listed on https://pypi.org/.

We also add all the Python files to the Docker image and define the entry point as `["python3", "mwe.py"]`. In this way, the command `python3 mwe.py` is automatically executed when the Docker image starts.

## Minimum Working Example - **`mwe.py`**
```py
if __name__ == "__main__":
    course_session = 'accounting_001'
    label_type = 'dropout'
    extract_data(course_session)
    build_course_dataset(course_session, label_type)
    build_predictive_model(course_session, label_type)
```

`mwe.py` is the main script responsible for executing all steps in the pipeline. Here are three primary steps:

1. `extract_data` executes SQL queries to extract student data from MORF's database and save the data into various CSV files.
1. `build_course_dataset` combines the CSV files generated from the previous step with the data labels, generating a complete CSV file that can be used for building ML models.
1. `build_predictive_model` performs model selection and hyperparameter tuning to identify the best model for predicting student dropout. It generates an `output.csv` file containing the prediction results, which get evaluated by MORF. The evaluation results are then sent to you via email.

**Notes 1**: File I/O is slow so if you are working with large datasets, you may want to avoid CSV reading/writing. Instead, you can have each data pipeline step return their output dataframes, then use them as input to the next step.

**Notes 2**: in the file `mwe.py` you will also see a function `build_dummy_output`, which generates an `output.csv` with random column values. This is useful when you want to submit a job to test earlier steps in the pipeline while you don't have a complete `output.csv` yet, as MORF will evaluate this dummy output file instead. If MORF sends you an email indicating that the job has completed, this means your current code is working fine.

### Extracting the Data - **`extract_data`**
```py
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
```

`extract_data` creates a new directory at `/temp-data/<course session name>` to store the extracted data. This structure is useful when you work with data from multiple course sessions, each of which will have its own directory. In this MWE, we only work with session `001` of the course `accounting`.

Next, we call `extract_coursera_sql_data`, which queries from three databases to get data about quiz submissions, lecture viewing and course grades. This step generates several CSV files, one from each SQL table. Finally, `combine_features` combines the generated CSV files into a single feature CSV; in this process, students who have missing data in any feature are excluded from analysis.

#### Extracting Data from Coursera **`extract_coursera_sql_data`**
```py
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
```
This function calls three sub-functions to query from three database tables. These functions follow the same logic, which is illustrated in `extract_quiz_submissions` below.

```py
def execute_mysql_query(query, db_name):
	"""
	Send a MySQL query to the specified course database.
	:param query: the query string.
	:param db_name: the database name, which is also the course session name, e.g., accounting_001
	:return the SQL output in JSON format.
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
```

As of MORF v0.1.0, you no longer need to initialize a MySQL session. Instead, SQL queries can be sent to the HTTP endpoint `http://course_database_proxy/sql`, as shown in `execute_mysql_query`. Then the HTTP response can be converted to JSON and to a Pandas DataFrame. Finally, this dataframe is stored in a CSV file to be used later.

Note that in the line
```
df = pd.DataFrame(list(response[3]), columns = response[:3])
```
the number 3 comes from the 3 columns being selected in the SQL query: `item_id, session_user_id` and `raw_score`. If your query selects 5 columns, you will replace 3 with 5 here.

#### Combining Features **`combine_features`**
```py
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
```
We expect the final feature dataset to be in wide format, where each student occupies one row and each feature occupies one column. However, the quiz submission and lecture viewings data are currently in long format, where each tuple (student, item id, value) occupies one row. Thus, we convert long-to-wide using Pandas' `.pivot_table`. In this way, the quiz submission data will transform from

| userID | item_id | raw_score |
| ------ | ---- | ----------- |
| a2f2ed432d514461136c895ab8bc47e23fd0011d | 0 | 7 |
| a2f2ed432d514461136c895ab8bc47e23fd0011d | 1 | 15 |
| a2f2ed432d514461136c895ab8bc47e23fd0011d | 2 | 23 |
| ...                                      | ... | ... |
| dba1c435cdfdaa383458560cf2969c404895abe8 | 1 | 9 |

to

| userID | quiz0_raw_score | quiz1_raw_score | quiz2_raw_score |
| ------ | --------------- | --------------- | --------------- |
| a2f2ed432d514461136c895ab8bc47e23fd0011d | 7 | 15 | 23 |
| ...                                      | ... | ... | ... |
| dba1c435cdfdaa383458560cf2969c404895abe8 | 0 | 9 | 0 |

A similar transformation is applied to the lecture viewing dataset. Thus, the quiz submission and lecture data will be compatible with the course grades dataset, which is already in wide format. As the final step, we concatenate the three datasets along the columns. Note the `join = "inner"` parameter specifies that the final student set should be the intersection (rather than the union) of the students in each dataset. In other words, students who do not appear in all three datasets are excluded. The rationale is that we cannot predict their dropout status if they have missing feature data.

### **`build_course_dataset`**
```py
TRAIN_PATH = "/morf-data/labels-train-with-underscore.csv"
TEST_PATH = "/morf-data/labels-test-with-underscore.csv"

def build_course_dataset(course_session, label_type):
    """
    Combine the feature dataframe with the label dataframe.
    :param course_session: the name of the course session.
    :param label_type: the type of label to predict.
    :return: a DataFrame with both feature columns and a label column.
    """
    df_labels = pd \
        .concat([pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)]) \
        .query("label_type == @label_type") \
        .set_index("userID") \
        .loc[:, ["label_value"]]

    df_all_data = pd \
        .read_csv("features.csv", index_col = "session_user_id") \
        .merge(df_labels, left_index = True, right_index = True)

    df_all_data.to_csv("all_data.csv", index = False)
```
Having the feature data in place, the next step is to grab the label values (i.e., which students ended up dropping out). At the time of this tutorial, the label data are not stored in any database, but in two CSV files that are pre-loaded to the Docker environment: `labels-train-with-underscore.csv` and `labels-test-with-underscore.csv`. Once we filter the label type in these dataframes to match our prediction goal, i.e., `label_type == 'dropout'`, we can merge them with our feature dataset to have both the feature and label data in one place.

**Notes**: the `userID` column in these two CSV files correspond to the `session_user_id` column in the SQL databases. The three data tables that we queried from earlier -- `quiz_submissions`, `lecture_viewings`, and `course_grades` -- already have `session_user_id`, so we can just perform a normal merge. Some other tables, such as [`forum_posts`](https://wiki.illinois.edu/wiki/display/coursera/forum_posts), may only have `user_id` instead of `session_user_id`. If you want to use these tables, during the SQL query step earlier you need to perform a join with [`hash_mapping`](https://wiki.illinois.edu/wiki/display/coursera/hash_mapping) to retrieve the corresponding `session_user_id` from the `user_id`. Example query:

```sql
SELECT thread_id, post_time, b.session_user_id
FROM forum_comments as a
LEFT JOIN hash_mapping as b
ON a.user_id = b.user_id
```

### **`build_predictive_model`**

The final step in our pipeline is to build an ML to predict student dropout. Note that the workflow can be as simple as (1) initialize a specific model, (2) split train/test, (3) train model and test. Here we present a more complex model selection procedure to improve the evaluation outcomes.

We start with three candidate models, each accompied by a grid of possible hyperparamter values. For models that involve randomness, we also specify the random state to support replicability.

```py
SEED = 1

candidate_models = [
    (LogisticRegression(random_state = SEED), {"C" : [1, 10, 100], "max_iter" : [100, 1000]}),
    (RandomForestClassifier(random_state = SEED), {"n_estimators" : [10, 50, 100], "max_depth" : [10, 20, 30]}),
    (KNeighborsClassifier(), {"n_neighbors" : [5, 10, 15], "weights" : ["uniform", "distance"]})
]
```

To select the best model and the associated best hyperparameter values, we perform the following steps:

1. Split the data into an outer train set and a test set.
1. Further split the outer train set into an inner train set and a validation set.
1. For each base model -- `LogisticRegression`, `RandomForestClassifier` and `KNeighborsClassifier` -- identify the associated best hyperparameter values via [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
1. Compare the three base models with their respective best hyperparameter sets on the validation data.
1. Fit the winner model on the outer train set.
1. Generate its predictions on the test set, stored in `/output/output.csv`.

The full code for the above procedure is as follows. 

```py
cv_inner = KFold(n_splits = 10, shuffle = True, random_state = SEED)


def select_best_model(X_train, X_val, y_train, y_val):
    """
    For each base estimator, perform grid-search CV to find the hyperparameters that work best on the train set.
    Select the base estimator and associated hyperparameters that work best on the validation set.
    :param X_train: the training feature dataset
    :param X_val: the validation feature dataset
    :param y_train: the training labels
    :param y_val: the validation labels
    """
    best_model = None
    best_val_score = 0
    for (base_estimator, param_grid) in candidate_models:
        # For each base model, identify the associated best hyperparameter values by using GridSearchCV
        search = GridSearchCV(base_estimator, param_grid, cv = cv_inner, scoring = 'f1')
        result = search.fit(X_train, y_train)
        best_estimator = result.best_estimator_

        # Compare the three base models with their respective best hyperparameter sets on the validation data.
        y_val_pred = best_estimator.predict(X_val)
        val_score = f1_score(y_val, y_val_pred)
        if val_score > best_val_score:
            best_model = best_estimator
    return best_model


def build_predictive_model(course_session, label_type):
    """
    Build a predictive model through model selection and hyperparameter tuning.
    Evaluate the model on a test dataset and save the prediction output to csv.
    :param course_session: the name of the course session.
    :param label_type: the type of label to predict.
    :return:
    """
    df_features = pd.read_csv("all_data.csv")
    X = df_features.drop(columns = "label_value").values
    y = df_features["label_value"].values

    # Split the data into an outer train set and a test set.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, stratify = y, random_state = SEED
    )

    # Further split the outer train set into an inner train set and a validation set.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, random_state = SEED
    )

    # Find the best model and best hyperparameters
    best_model = select_best_model(X_train, X_val, y_train, y_val)

    # Fit the winner model on the outer train set.
    best_model.fit(X_train_val, y_train_val)

    # Generate its predictions on the test set, stored in /output/output.csv.
    y_pred = best_model.predict(X_test)
    y_score = best_model.predict_proba(X_test)
    output = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred, 'y_score': y_score[:, 1]})
    output.to_csv('/output/output.csv', index = False)
```

## After job execution
MORF will then evaluate the test performance based the generated `output.csv` and send the results via email. With the code in this MWE, here is the expected evaluation result:

| accuracy | cohen_kappa | N | N_n | N_p | auc | log_loss | precision | recall | f1_score | specificity |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.8755414862 | 0.7509067872 | 6002 | 2988 | 3014 | 0.9309881086 | 0.763262444 | 0.8246920653 | 0.9552090246 | 0.8851652575 | 0.9462365591 |