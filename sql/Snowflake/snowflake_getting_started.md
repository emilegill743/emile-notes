# Snowflake Docs - Getting Started

https://docs.snowflake.com/en/user-guide-getting-started.html

## SnowSQL

*CLI Client for connecting to Snowflake.*

- [Installation](https://docs.snowflake.com/en/user-guide/snowsql-install-config.html)
- [Connecting](https://docs.snowflake.com/en/user-guide/snowsql-start.html)
    - **SnowSQL** - `snowsql -a <account_name> -u <username>`
    - **Web UI** - `https://<account_name>.snowflakecomputing.com`

## Creating Objects

### Creating a Database
```sql
-- Create Database
create or replace database sf_tuts;

-- Checking Current Context
select current_database(), current_schema();
```

### Creating a Table
```sql
-- Create Table
create or replace table emp_basic (
    first_name string ,
    last_name string ,
    email string ,
    streetaddress string ,
    city string ,
    start_data date
);
```

### Creating a Virtual Warehouse
```sql
-- Create Virtual Warehouse
create or replace warehouse sf_tuts_wh with
    warehouse_size='X-SMALL'
    auto_suspend=180
    auto_resume=true
    initially_suspended=true;

-- Checking Current Context
select current_warehouse();
```

## Stage Data Files

Snowflake supports loading data from files staged in either an internal (Snowflake) stage or external (Amazon S3, Google Cloud Storage, or Microsoft Azure).

### Uploading local data
```bash
put file://C:\temp\employees0*.csv @sf_tuts.public.%emp_basic;
```

- `file:` specifies the full directory path and names of the files on your local machine to stage. Note that file system wildcards are allowed.

- `@<namespace>.%<table_name>` indicates to use the stage for the specified table, in this case the emp_basic table.

### List Staged Files
```sql
list @sf_tuts.public.%emp_basic;
```

## Copy Data into Table
```sql
-- Copy Date into Table from Stage
copy into emp_basic
    from @%emp_basic
    file_format = (type = csv field_optionally_enclosed_by='"')
    pattern = '.*employees0[1-5].csv.gz'
    on_error = 'skip_file';
```

- `FILE_FORMAT` specifies the file type as CSV, with double quotes (`"`) used to enclose strings. 
  - [Supported file formats](https://docs.snowflake.com/en/sql-reference/sql/create-file-format.html)
- `PATTERN` applies pattern matching to load data from al files matching a regular expression.
- `ON_ERROR` specify what to do when an error is encountered, by default the command will stop when an error is detected.
  - `COPY` command also provides options for validating files before they are loaded ([validating staged files](https://docs.snowflake.com/en/sql-reference/sql/copy-into-table.html#validating-staged-files)).

## Query/Manipulate the Loaded Data

### Query all Data
```sql
select * from emp_basic;
```

### Insert Additional Rows
```sql
insert into emp_basic values
  ('Clementine','Adamou','cadamou@sf_tuts.com','10510 Sachs Road','Klenak','2017-9-22') ,
  ('Marlowe','De Anesy','madamouc@sf_tuts.co.uk','36768 Northfield Plaza','Fangshan','2017-1-26');
```

### Query Rows based on Condition
```sql
select email from emp_basic where email like '%.uk';

select first_name, last_name, DATEADD('day',90,start_date) FROM emp_basic WHERE start_date <= '2017-01-01';
```