# Introduction to Airflow

- [Introduction to Airflow](#introduction-to-airflow)
  - [Introduction to Airflow](#introduction-to-airflow-1)
    - [What is Airflow?](#what-is-airflow)
    - [DAGs](#dags)
  - [Implementing Airflow DAGs](#implementing-airflow-dags)
    - [Airflow Operators](#airflow-operators)
    - [Bash Operator](#bash-operator)
    - [Airflow Tasks](#airflow-tasks)
    - [Additional Operators](#additional-operators)
    - [Airflow Scheduling](#airflow-scheduling)
  - [Maintaining and monitoring Airflow workflows](#maintaining-and-monitoring-airflow-workflows)
    - [Airflow Sensors](#airflow-sensors)
    - [Airflow Executors](#airflow-executors)
    - [Debugging and troubleshooting Airflow](#debugging-and-troubleshooting-airflow)
    - [SLAs and reporting in Airflow](#slas-and-reporting-in-airflow)
  - [Building production pipelines in Airflow](#building-production-pipelines-in-airflow)
    - [Working with templates](#working-with-templates)
    - [Branching](#branching)
    - [Creating a production pipeline](#creating-a-production-pipeline)

## Introduction to Airflow

### What is Airflow?

> **Workflow**: A set of steps to accomplish a given data engineering task, such as downloading files, copying data, filtering information, writing to a database etc.

**Apache Airflow** is a platform to program workflows, including: creation, scheduling and monitoring. Workflows are impletemented as DAGs (Directed Acyclic Graphs), authored in Python. Airflow may be accessed by code, CLI or via a web interface.

### DAGs

- Represent the set of tasks (operators, sensors etc.) that make up a workflow.
- Consist of tasks and the dependencies between tasks.
- Created with various details about the DAG, including name, start date, owner etc.
- Written in Python, but can use components written in other languages.

```python
from airflow.models import DAG

from datetime import datetime

default_arguments = {
    'owner': 'emilegill',
    'email': 'emilegill@email.com',
    'start_date': datetime(2020, 1, 20),
    'retries': 3
}

etl_dag = DAG(
    dag_id='etl_workflow',
    default_args=default_args
)
```

- **Directed:** Inherent flow representing dependencies between components.
- **Acyclic:**: Does not loop/cycle/repeat
- **Graph:** Actual set of components

**DAGs on the command line:**

`airflow` CLI contains numerous subcommands, many of which are related to DAGs:

`airflow list_dags` : show all recognised DAGs

`airflow run <dag_id> <task_id> <start_date>` Run a specific task from a DAG

## Implementing Airflow DAGs

### Airflow Operators

The most common type of task in Airflow is the `Operator`.

- Represent a single task in a workflow, e.g. running a command, sending an email or running a Python script.
- Usually run independently, i.e. all resources needed to complete the task are contained within the operator.
- Do not share information between each other (although XComs do make this possible if neccessary).
- Various Operators to perform different tasks, e.g. `BashOperator`, `DummyOperator`, `PythonOperator`.

### Bash Operator

- Executes a given Bash command or script.
- Runs the command in a temporary directory, which is automatically cleaned up afterwards.
- Can specify environment variables for the command.

```python
from airflow.operators.bash_operator import BashOperator

example_task = BashOperator(task_id='bash_example',
                            bash_command='echo "Example!"',
                            dag=ml_dag)

shell_task = BashOperator(task_id='bash_example',
                         bash_command='runcleanup.sh',
                         dag=ml_dag)

bash_task = BashOperator(task_id='clean_addresses',
                         bash_command='cat addresses.txt | awk "NF==10" > cleaned.txt',
                         dag=ml_dag)
```

**Considerations:**

- Not guaranteed to run in the same location/environment.
- May require use of environment variables, e.g. AWS credentials, database connection details etc.
- Can be difficult to run jobs with elevated privileges, access to resources must be set up for the specific user running the task.

### Airflow Tasks

- Instances of Operators.
- Usually assigned to a variable in Python.
- Referred to by its `task_id`.

```python
example_task = BashOperator(task_id='bash_example',
                            bash_command='echo "Example!"'),
                            dag=dag)
```

**Task dependencies:**

- Define a given order of task completion.
- Not required for a given workflow, but present in most.
- Referred to as upstream or downstream tasks.
- Defined by **bitshift** operators: `>>` $\implies$ upstream (before), `<<` $\implies$ downstream task (after).

```python
# Define tasks
task1 = BashOperator(task_id='first_task',
                     bash_command='echo 1',
                     dag=example_dag)

task2 = BashOperator(task_id='second_task',
                     bash_command='echo 2',
                     dag=example_dag)

# Set task1 to run before task2
task1 >> task2 # or task2 << task1
```

### Additional Operators

**PythonOperator:**

- Executes a Python function/callable.

```python
from airflow.operators.python_operator import PythonOperator

def printme():
    print("This goes in the logs!")

python_task = PythonOperator(
    task_id='simple_print',
    python_callable=printme,
    dag=example_dag
)
```

- Can pass in arguments to the Python code.

```python
def sleep(length_of_time):
    time.sleep(length_of_time)

sleep_task = PythonOperator(
    task_id='sleep',
    python_callable=sleep,
    op_kwargs={'length_of_time': 5},
    dag=example_dag
)
```

**EmailOperator:**

- Sends an email
- Can contain typical components:
  - HTML content
  - Attachments

```python
from airflow.operators.email_operator import EmailOperator

email_task = EmailOperator(
    task_id='email_sales_report',
    to='sales_manager@example.com',
    subject='Automated Sales Report',
    html_content='Attached is the latest sales report',
    files='latest_sales.xlsx',
    dag=example_dag
)
```

### Airflow Scheduling

**DAG Runs**: 

- A specific instance of a workflow at a point in time.
- Can be run manually or bu `schedule_interval`
- Maintain state for each workflow and tasks within: `running`, `failed`, `success`, `queued`, `skipped`.

**Schedule details:**

- `start_date`: date/time to initially schedule DAG run.
- `end_date` (optional): date/time to stop running new DAG instances.
- `max_retries` (optional): how many attempts to make in running a DAG.
- `schedule_interval`: how often to run a DAG

**Schedule interval syntax:**:

The `schedule_interval` parameter may be specfied using `cron` syntax:

```
# ┌───────────── minute (0 - 59)
# │ ┌───────────── hour (0 - 23)
# │ │ ┌───────────── day of the month (1 - 31)
# │ │ │ ┌───────────── month (1 - 12)
# │ │ │ │ ┌───────────── day of the week (0 - 6) (Sunday to Saturday;
# │ │ │ │ │                                   7 is also Sunday on some systems)
# │ │ │ │ │
# │ │ │ │ │
# * * * * * <command to execute>
```

e.g.

```bash
0 12 * * *              # Run daily at noon

* * 25 2 *              # Run once per minute on February 25th

0,15,30,45 * * * *      # Run every 15minutes
```

Or, also, may be defined as one of Airflow`s preset values: `@hourly`, `@daily`, `@weekly`.

Special presets:
- `None` $\implies$ Don't schedule ever, use only with manual triggers.
- `@once` $\implies$ schedule only once.

**Note:** When scheduling a DAG, Airflow will wait one `schedule_interval` from it's `start_date` before triggering the DAG for the first time. Therefore, a DAG with

```python
'start_date': datetime(2020, 2, 25),
'schedule_interval': @daily
```
will trigger for the first time on February 26th 2020.

## Maintaining and monitoring Airflow workflows

### Airflow Sensors

**Sensor:**

- Operator that waits for a certain condition to be true.
  - Creation of a file
  - Upload of a database record
  - Certain response from a web request
- Can define how often to check for a condition to be true
- Assigned to tasks
- Derived from `airflow.sensors.base_sensor_operator`

**Sensor arguments:**

- `mode`: How to check for the condition
  - `mode='poke'` (default): Run repeatedly and maintain task slot
  - `'mode='reschedule'`: Give up task slot an try again later
-  `poke_interval`: How often to wait between checks.
-  `timeout`: How long to wait before failing task.

**File Sensor:**
```python
from airflow.contrib.sensors.file_sensors import FileSensor

file_sensor_task = FileSensor(task_id='file_sense',
                              filepath='salesdata.csv',
                              poke_interval=300,
                              dag=sales_report_dag)

init_sales_cleanup >> file_sensor_task >> generate_report
```

**Other Sensors:**

- `ExternalTaskSensor`: wait for a task in another DAG to complete.
- `HttpSensor`: Request a web URL and check content.
- `SQLSensor`: Runs a SQL query to check for content.
- Many others in `airflow.sensors` and `airflow.contrib.sensors`.

### Airflow Executors

Executors run tasks in Airflow, different executors handle running of tasks differently.

- `SequentialExecutor`(default):
  - Runs one task at a time
  - Useful for debugging, but not recommended for production
  
- `LocalExecutor`:
  - Runs on a single system
  - Treats tasks as processes
  - *Parallelism* is defined by the user as either unlimited, or limited to a certain number of concurrent tasks.
  - Can utilise all resources of the host system
  
- `CeleryExecutor`:
  - **Celery**: a queuing system written in Python, allowing multiple systems to communicate as a cluster
  - Uses Celery backend as a task manager
  - Multiple worker systems may be defined
  - Significantly more difficult to setup and configure
  - Powerful method for organisations with extensive workflows

The executor being used by an Airflow configuration can be found in its `airflow.cfg` file, under the entry `executor=`.

### Debugging and troubleshooting Airflow

**DAG won't run on schedule:**

- Check if scheduler is running
  - We can start scheduler from the command line `airflow scheduler`
- At least one `schedule_interval` hasn't passed since the `start_date`.
  - Modify the attributes to meet requirements
- Not enough tasks free within executor to run
  - Change executor type
  - Add systems/system resources (RAM, CPUs etc.)
  - Change DAG scheduling

**DAG won't load (not in web UI/`airflow list_dags`):**

- Verify DAG file is in correct folder
- Determine DAGs folder specified in `airflow.cfg`
- Syntax errors
  - `airflow list_dags` (errors will be listed in output)
  - `python3 <dagfile.py>` (won't run the actual code within the DAGs but will pick up on any syntax errors)

### SLAs and reporting in Airflow

**SLA**: *Service Level Agreement*

- Within Airflow this is the time a DAG or task should require to run.
- And SLA miss is any time a task/DAG does not meet this expected timing.
- On an SLA miss an email is sent out and a log stored.
- SLA misses can also be viewed in the web UI under `Browse:SLA Misses`

**Defining SLAs:**

- Using the `sla` argument on a task definition:

```python
task1 = BashOperator(task_id='sla_misses',
                     bash_command='runcode.sh',
                     sla=timedelta(seconds=30),
                     dag=dag) 
```

- On the `default_args` dictionary:

```python
default_args = {
  'sla': timedelta(minutes=20),
  'start_date': datetime(2020, 2, 20)
}

dag = DAG('sla_dag', default_args=default_args)
```

**General Reporting:**

- Options for success, failure, retry defined as keys in `default_args` dictionary.

```python
default_args = {
  'email': ['emile@emilegill.com'],
  'email_on_failure': True,
  'email_on_retry': False,
  'email_on_success': True
}
```

## Building production pipelines in Airflow

### Working with templates

**Templates**:
- Allow for substituting information during a DAG run
- Provide added flexibility when defining tasks
- Created using `Jinja` templating language.

**Non-templated example:**

```python
t1 = BashOperator(
      task_id='first_task',
      bash_command='echo "Reading file1.txt"',
      dag=dag)

t1 = BashOperator(
      task_id='second_task',
      bash_command='echo "Reading file2.txt"',
      dag=dag)
```

**Templated example:**

Simple example-
```python
templated_command="""
  echo "Reading {{params.filename}}"
"""

t1 = BashOperator(task_id='template_task',
                  bash_command=templated_command,
                  params={'filename': 'file1.txt'},
                  dag=example_dag)

t1 = BashOperator(task_id='template_task',
                  bash_command=templated_command,
                  params={'filename': 'file2.txt'},
                  dag=example_dag)
```

More advanced example-
```python
templated_command = """
  {% for filename in params.filenames %}
    echo "Reading {{filename}}"
  {% endfor %}
"""

t1 = BashOperator(task_id='templated_task',
                  bash_command=templated_command,
                  params={'filenames': ['file1.txt', 'file2.txt']},
                  dag=example_dag)
```

**Variables:**

Airflow provides several built-in runtime variables, providing infor about DAG runs, tasks, system config etc.

Examples:

| description                                    | variable             |
| ---------------------------------------------- | -------------------- |
| Execution Date (YYYY-MM-DD)                    | {{ ds }}             |
| Execution Date, no dashes (YYYYMMDD)           | {{ds_nodash}}        |
| Previous Execution Date (YYYY-MM-DD)           | {{ prev_ds }}        |
| Previous Execution Date, no dashes (YYYYMMDD)  | {{ prev_ds_nodash }} |
| DAG object:                                    | {{ dag }}            |
| Airflow config object:                         | {{ conf }}           |

**Macros:**

In addition to the other variables provided, airflow also provides a `macros` variable referencing the Airflow macros package. This provides various useful objects/methods for Airflow templates.

Examples:

| macro | description |
| ---------------------------------------------- | -------------------- |
| {{ macros.datetime }} | `datetime.datetime` object |
| {{ macros.timedelta }} | `datetime.timedelta` object |
| {{ macros.uuid }} | Python's `uuid` object |
| {{ macros.ds_add('2020-04-15', 5)  }}  | Modify days from a date, e.g. this will return 2020-04-20 |

### Branching

- Provides conditional logic, i.e. tasks which can be selectively executed or skippedon the result of an operator.

```python
def branch_test(**kwargs):
  if int(kwargs['ds_no_dash']) % 2 == 0:
    return 'even_day_task'
  else:
    return 'odd_day_task'

branch_task = BranchPythonOperator(task_id='branch_task',
                                   provide_context=True,
                                   python_callable=branch_test,
                                   dag=dag)

start_task >> branch_task >> even_day_task >> even_day_task2
branch_task >> odd_day_task >> odd_day_task2
```
- Here the even_day_task and odd_day tasks will be run depending on the condition defined in our `branch_test` function.

### Creating a production pipeline

**Running tasks and DAGs:**

```bash
# Run a specific task from the command line
airflow run <dag> <task_id> <date>

# Run a full DAG
airflow trigger_dag -e <date> <dag_id>
```

**Pipeline Example:**

```python
from airflow.models import DAG
from airflow.contrib.sensors.file_sensor import FileSensor
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators.python_operator import BranchPythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.email_operator import EmailOperator
from dags.process import process_data
from datetime import datetime, timedelta

def process_data(**context):
  file = open('/home/repl/workspace/processed_data.tmp', 'w')
  file.write(f'Data processed on {date.today()}')
  file.close()

default_args = {
  'start_date': datetime(2019,1,1),
  'sla': timedelta(minutes=90)
}
    
dag = DAG(dag_id='etl_update', default_args=default_args)

sensor = FileSensor(task_id='sense_file', 
                    filepath='/home/repl/workspace/startprocess.txt',
                    poke_interval=45,
                    dag=dag)

bash_task = BashOperator(task_id='cleanup_tempfiles', 
                         bash_command='rm -f /home/repl/*.tmp',
                         dag=dag)

python_task = PythonOperator(task_id='run_processing', 
                             python_callable=process_data,
                             provide_context=True,
                             dag=dag)


email_subject="""
  Email report for {{ params.department }} on {{ ds_nodash }}
"""


email_report_task = EmailOperator(task_id='email_report_task',
                                  to='sales@mycompany.com',
                                  subject=email_subject,
                                  html_content='',
                                  params={'department': 'Data subscription services'},
                                  dag=dag)


no_email_task = DummyOperator(task_id='no_email_task', dag=dag)


def check_weekend(**kwargs):
    dt = datetime.strptime(kwargs['execution_date'],"%Y-%m-%d")
    # If dt.weekday() is 0-4, it's Monday - Friday. If 5 or 6, it's Sat / Sun.
    if (dt.weekday() < 5):
        return 'email_report_task'
    else:
        return 'no_email_task'
    
    
branch_task = BranchPythonOperator(task_id='check_if_weekend',
                                   provide_context=True,
                                   python_callable=check_weekend,
                                   dag=dag)

    
sensor >> bash_task >> python_task

python_task >> branch_task >> [email_report_task, no_email_task]






