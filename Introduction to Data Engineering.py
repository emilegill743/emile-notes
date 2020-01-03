###INTODUCTION TO DATA ENGINEERING###

'''What is data engineering?'''

    #Definition: An engineer that develops, constructs, tests and maintains architechtures such as databases and large-scale processing systems.
        #Processing large amounts of data
        #Use of clusters of machines
    
    #Responsibilities:
        #Develop scalable data architechture
        #Streamline data acquisition
        #Set up processes to bring together data
        #Clean corrupt data
        #Well versed in cloud technology

'''Tools of the data engineer'''

    #Databases
        #e.g. SQL, NoSQL
    
    #Processing
        #Clean
        #Aggregate
        #Join

        #e.g. Spark, Hive

    #Scheduling
        #Plan jobs with specific intervals
        #Resolve dependency requirements of jobs

        #e.g. Airflow, oozie,  cron
    
'''Cloud providers'''

    #AWS
        #Storage - AWS S3
        #Computation - AWS EC2
        #Databases - AWS RDS

    #Azure
        #Storage - Azure Blob Storage
        #Computation - Azure Virtual Machines
        #Databases - Azure SQL Database

    #Google Cloud
        #Storage - Google Cloud Storage
        #Computation - Google Compute Engine
        #Databases - Google Cloud SQL

'''Databases'''

    #Hold data
    #Organise data
    #Retrieve/Search data through DBMS

    #SQL vs NoSQL

'''Parallel Computing'''
    
    #Split task into subtasks
    #Distributed over several computers

    #Benefits:
        #Greater processing power
        #Memory- partition the dataset

    #Risks:
        #Overhead due to communication
            #Task need to be large
            #Need several processing units

    #MultiProcessing
        from multiprocessing import Pool

        def take_mean_age(year_and_group):
            year, group = year_and_group
            return(pd.Dataframe({"Age" : group["Age"].mean()}, index=[year]))
        
        with Pool(4) as p:
            results = p.map(take_mean_age, athlete_events.groupby("Year"))
            #Splits tasks over 4 cores

        results_df = pd.concat(results)

    #Dask

        #Performs groupby, apply and multiprocessing out of the box

        import dask.dataframe as dd

        #Partition dataframe into 4
        athlete_events_dask = dd.from_pandas(athlete_events, npartitions = 4)
        
        #Run parallel computations on each partition
        results_df = athlete_events_dask.groupby('Year').Age.mean().compute()


    # Function to apply a function over multiple cores
        
        #Using multiprocessing.Pool
            @print_timing
            def parallel_apply(apply_func, groups, nb_cores):
                with Pool(nb_cores) as p:
                    results = p.map(apply_func, groups)
                return pd.concat(results)

            # Parallel apply using 1 core
            parallel_apply(take_mean_age, athlete_events.groupby('Year'), 1)

            # Parallel apply using 2 cores
            parallel_apply(take_mean_age, athlete_events.groupby('Year'), 2)

            # Parallel apply using 4 cores
            parallel_apply(take_mean_age, athlete_events.groupby('Year'), 4)

        #Using dask

            import dask.dataframe as dd

            # Set the number of pratitions
            athlete_events_dask = dd.from_pandas(athlete_events, npartitions = 4)

            # Calculate the mean Age per Year
            print(athlete_events_dask.groupby('Year').Age.mean().compute())

'''Parallel computation frameworks'''

    #Hadoop
        #HDFS - Distributed file system
        #Map Reduce - Big data processing paradigm

    #Hive
        #Runs on Hadoop
        #Structure Query Language: Hive SQL
        #Initially MapReduce, now integrates with several other data processing tools

    #Spark
        #Relies on RDDs (resilient distributed datasets)
        #Similar to list of tuples
        #Trasformations .map() or .filter()
        #Actions .count() or .first()
        #PySpark - Python interface to Spark
        #DataFrame abstraction - looks similar to pandas
        #E.g.
            (athlete_events_spark
                .groupby('Year')
                .mean('Age')
                .show())

            # Print the type of athlete_events_spark
            print(type(athlete_events_spark))

            # Print the schema of athlete_events_spark
            print(athlete_events_spark.printSchema())

            # Group by the Year, and find the mean Age
            print(athlete_events_spark.groupBy('Year').mean('Age'))

            # Group by the Year, and find the mean Age
            print(athlete_events_spark.groupBy('Year').mean('Age').show())

            #Running Spark file (bash)
            spark-submit \
            --master local[4] \
            /home/repl/spark-script.py

'''Workflow Scheduling Frameworks'''

    #DAG - Directed Acyclic Graph
        #Set of Nodes
        #Directed Edges
        #No cycles

    #Tools
        #Linux cron
        #Luigi (Spotify)
        #Apache Airflow (Airbnb)

    #Airflow
        #Examples
        #Simple
            from airflow.models import DAG
            dag = DAG(dag_id="sample",
                    ...,
                    schedule_interval="0 0 * * *")

            #cron expression
                # minute (0-59)
                # hour (0-23)
                # day of month (1-31)
                # month (1-12)
                # day of week (0-6)
        #1
            #Create the DAG object
            dag = DAG(dag_id="example_dag", ... , schedule_interval="0****")

            #Define operations
            start_cluster = StartClusterOperator(task_id="start_cluster", dag=dag)
            ingest_customer_data = SparkJobOperator(task_id="ingest_customer_data", dag=dag)
            ingest_product_data = SparkJobOperator(task_id="ingest_product_data", dag=dag)
            enrich_customer_data = PythonOperator(task_id="enrich_customer_data", ... , dag=dag)

            #Set up dependency flow
            start_cluster.set_downstream(ingest_customer_data)
            ingest_customer_data.set_downstream(enrich_customer_data)
            ingest_product_data.set_downstream(enrich_customer_data)
        #2
            # Create the DAG object
            dag = DAG(dag_id="car_factory_simulation",
                    default_args={"owner": "airflow","start_date": airflow.utils.dates.days_ago(2)},
                    schedule_interval="0 * * * *")

            # Task definitions
            assemble_frame = BashOperator(task_id="assemble_frame", bash_command='echo "Assembling frame"', dag=dag)
            place_tires = BashOperator(task_id="place_tires", bash_command='echo "Placing tires"', dag=dag)
            assemble_body = BashOperator(task_id="assemble_body", bash_command='echo "Assembling body"', dag=dag)
            apply_paint = BashOperator(task_id="apply_paint", bash_command='echo "Applying paint"', dag=dag)

            # Complete the downstream flow
            assemble_frame.set_downstream(place_tires)
            assemble_frame.set_downstream(assemble_body)
            assemble_body.set_downstream(apply_paint)

            #DAG:

                                #-->place_tires
            #assemble_frame -->
                                #-->assemble_body --> apply_paint

'''Extract'''

    #Persistent storage (not suited for data processing) --> Memory
    #E.g. file on S3, database, API

    #Text files
        #Unstructured
        #Flat files - row = record, column = attribute
            #E.g. .txv, .csv
    
    #JSON - JavaScript Object Notation
        #Semi-structured
        #Four atomic datatypes
            #number
            #string
            #boolean
            #null
        #Two composite data types
            #array
            #object
        #Comparative to dictionaries in Python
        #json --> dict
            import json
            result = json.loads('{"key_1" : "value_1",
                                  "key_2" : "value_2"}')

            print(results["key_1"])

    #Data on the Web
        #Request page --> Server response
        
        #API - application programming interface
        #Send data as JSON
        #E.g. Twitter API
        #Example web-request
            import requests

            # Fetch the Hackernews post
            resp = requests.get("https://hacker-news.firebaseio.com/v0/item/16222426.json")

            # Print the response parsed as JSON
            print(resp.json())

            # Assign the score of the test to post_score
            post_score = resp.json()["score"]
            print(post_score)

    #Data in databases

        #Application databases
            #Optimised for transactions - insert or change or rows
            #OLTP - online transaction processes

        #Analytical databases
            #OLAP - online analytical processing
            #Column oriented
        
        #Extracting from databases

            #Connection string/URI
                'postgresql://[user[:password]@][host][:port]'
            
            #Use in Python
                #1
                    import sqlalchemy
                    connection_uri = "postresql://repl:password@localhost:5432/paglia"
                    db_engine = sqlalchemy.create_engine(connection_uri)

                    import pandas as pd
                    pd.read_sql("SELECT * FROM customer", db_engine)
                #2
                    # Function to extract table to a pandas DataFrame
                    def extract_table_to_pandas(tablename, db_engine):
                        query = "SELECT * FROM {}".format(tablename)
                        return pd.read_sql(query, db_engine)

                    # Connect to the database using the connection URI
                    connection_uri = "postgresql://repl:password@localhost:5432/pagila" 
                    db_engine = sqlalchemy.create_engine(connection_uri)

                    # Extract the film table into a pandas DataFrame
                    extract_table_to_pandas("film", db_engine)

                    # Extract the customer table into a pandas DataFrame
                    extract_table_to_pandas("customer", db_engine)

'''Tranform'''

    #Kind of transformation
        #Selection of attribute (e.g. 'email')
        #Translation of code values (e.g. 'New York' -> 'NY')
        #Data validation (e.g. date input in 'created_at')
        #Splitting columns into multiple columns
        #Joining from multiple sources

    #Example - split email into username and domain

        customer_df #Pandas DataFrae with customer data

        #Split email column into 2 columns on the '@' symbol
        split_email = customer_df.email.str.split("@", expand=True)
        #split_email will now consist of two columns, one with first half, one with second half

        #Create 2 new columns using resulting df
        customer_df = customer_df.assign(
            username = split_email[0],
            domain = split_email[1],
            )

    #Example 2 - Transforming in PySpark

        import pyspark.sql

        spark = pyspark.sql.SparkSession.builder.getOrCreate()

        spark.read.jdbc("jdbc@postgresql://localhost:5432/pagila",
                        "customer",
                        properties={"user" : "repl", "password" : "password"})

    #Example 3 - Joining in PySpark
        
        #Groupby ratings
        ratings_per_customer = ratings_df.groupBy("customer_id").mean("rating")

        #Join on customer ID
        customer_df.join(
            ratings_per_customer,
            customer_df.customer_id == ratings_per_customer.customer_id
        )

'''Load'''

    #Analytic DB
        #Aggregate queries
        #Online analytical processing (OLAP)
        #Column oriented
            #Queries about subsets of columns
            #Lend themselves better to parallelisation

    #Application DB
        #Lots of transactions
        #Online transactional processing (OLTP)
        #Row Oriented
            #Stored per record
            #Added per transaction

    #MPP Databases- Massive Parallel Processing Databases
        #Column oriented databases optimised for analytics
        #Run in distributed fashion
        #Queries split into sub-tasks and distributed across nodes
        #E.g.
            #Amazon Redshift
            #Azure SQL Data Warehouse
            #Google Big Query

    #Load from file to columnar storage format

        #Pandas .to_parquet method
        df.to_parquet("./s3://path/to/bucket/customer.parquet")
        #Pyspark .write.parquet() method
        df.write.parquet("./s3://path/to/bucket/customer.parquet")
        
        ### Amazon Redshift
        COPY customer_df
        FROM './s3://path/to/bucket/customer.parquet'
        FORMAT as parquet

    #Load to PostgreSQL
        pandas.to_sql()

        #Tranformation on data
        recommendations = transform_find_recommendations(ratings_df)

        #Load into PostgreSQL
        recommendations.to_sql("recommendations",
                                db_engine,
                                schema = "store"
                                if_exists= "replace")
        #if_exists can also be fail, append
        
'''Putting it all together'''

    #ETL function

        def extract_table_to_df(tablename, db_engine):
            return pd.read_sql("SELECT * FROM {}".format(tablename), db_engine)

        def split_columns_transform(df, column, pat, suffixes):
            #Converts column into str and splits it on pat

        def load_df_into_dwh(film_df, tablename, schema, db_engine):
            return pd.to_sql(tablename, db_engine, schema=schema, if_exist="replace")

        db_engines = {...} #confgure db engines
        def etl():
            #Extract
            film_df = extract_table_to_df("film", db_engines["store"])
            #Transform
            film_df = split_columns_transform(film_df, "rental_rate", ".", ["_dollar", "_cents"])
            #Load
            load_df_into_dwh(film_df, "film", "store", db_engines["dwh"])
        
    #Airflow Workflow Schedule

        from airflow.models import DAG
        from airflow.operators.python_operator import PythonOperator

        dag = DAG(dag_id="etl_pipeline",
                schedule_interval = "0 0 * * *")

        etl_task = PythonOperator(task_id = "etl_task",
                                python_callable = etl,
                                dag = dag)

        etl_task.set_upstream(wait_for_this_task)

        #Save as python file in dag folder of airflow
        # etl_dag.py --> ~/airflow/dags/

'''Case Study: Course Ratings'''

    #ETL Process
    #SQL_DB => Cleaning => Calculate Recommendations =>SQL_DataWareHouse
    
    #Querying table
        # Complete the connection URI
        connection_uri = "postgresql://repl:password@localhost:5432/datacamp_application"
        db_engine = sqlalchemy.create_engine(connection_uri)

        # Get user with id 4387
        user1 = pd.read_sql("SELECT * FROM rating WHERE user_id=4387", db_engine)

        # Get user with id 18163
        user2 = pd.read_sql("SELECT * FROM rating WHERE user_id=18163", db_engine)

        # Get user with id 8770
        user3 = pd.read_sql("SELECT * FROM rating WHERE user_id=8770", db_engine)

        # Use the helper function to compare the 3 users
        print_user_comparison(user1, user2, user3)

    #Average Rating per Course
            # Complete the transformation function
            def transform_avg_rating(rating_data):
            # Group by course_id and extract average rating per course
            avg_rating = rating_data.groupby('course_id').rating.mean()
            # Return sorted average ratings per course
            sort_rating = avg_rating.sort_values(ascending=False).reset_index()
            return sort_rating

            # Extract the rating data into a DataFrame    
            rating_data = extract_rating_data(db_engines)

            # Use transform_avg_rating on the extracted data and print results
            avg_rating_data = transform_avg_rating(rating_data)
            print(avg_rating_data) 

    #Filter out corrupt data
        course_data = extract_course_data(db_engines)

        # Print out the number of missing values per column
        print(course_data.isnull().sum())

        # The transformation should fill in the missing values
        def transform_fill_programming_language(course_data):
            imputed = course_data.fillna({"programming_language": "r"})
            return imputed

        transformed = transform_fill_programming_language(course_data)

        # Print out the number of missing values per column of transformed
        print(transformed.isnull().sum())
    
    #Using the recommender transformation
        # Complete the transformation function
        def transform_recommendations(avg_course_ratings, courses_to_recommend):
            # Merge both DataFrames
            merged = courses_to_recommend.merge(avg_course_ratings) 
            # Sort values by rating and group by user_id
            grouped = merged.sort_values("rating", ascending = False).groupby('user_id')
            # Produce the top 3 values and sort by user_id
            recommendations = grouped.head(3).sort_values("user_id").reset_index()
            final_recommendations = recommendations[["user_id", "course_id","rating"]]
            # Return final recommendations
            return final_recommendations

        # Use the function with the predefined DataFrame objects
        recommendations = transform_recommendations(avg_course_ratings, courses_to_recommend)  

    #Scheduling Daily Jobs
        rexommendations.to_sql(
                "recommendations",
                db_engine,
                if_exists="append"
        )

        def etl(db_engines):
            #Extract the data
            courses = extract_course_data(db_engines)
            rating = extract_rating_data(db_engines)
            #Clean up courses
            courses = transform_fill_programming_language(courses)
            #Get the average course ratings
            avg_course_rating = transform_courses_to_recommend(
                rating,
                courses)
            #Calculate the recommendations
            recommendations = transform_recommendations(
                avg_course_ratings,
                courses_to_recommend)
            #Load the recommendations into the database
            load_to_dwh(recommendations, db_engine, if_exists="replace")

        #Creating the DAG
        from airflow.models import DAG
        from airflow.operators.python_operator import PythonOperator

        dag = DAG(dag_id="recommendations"
                  scheduled_interval = "0 0 * * *")
        
        task_recommendations = PythonOperator(
            task_id = "recommendations_task",
            python_callable = etl
        )

    #Querying the recommendations
        def recommendations_for_user(user_id, threshold=4.5):
            # Join with the courses table
            query = """
            SELECT title, rating FROM recommendations
                INNER JOIN courses ON courses.course_id = recommendations.course_id
                WHERE user_id=%(user_id)s AND rating>%(threshold)s
                ORDER BY rating DESC
            """
            # Add the threshold parameter
            predictions_df = pd.read_sql(query, db_engine, params = {"user_id": user_id, 
                                                                    "threshold": threshold})
            return predictions_df.title.values

        # Try the function you created
        print(recommendations_for_user(12, 4.65))







