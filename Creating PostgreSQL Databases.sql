-----CREATING POSTGRESQL DATABASES-----

--CREATING A DATABASE--

    CREATE DATABASE db_name;

--CREATING TABLES--

    --TABLE
        --Variable no. rows
        --Fixed number of columns
        --Columns have specific data type
        --Each row is a record

    CREATE TABLE table_name (
        column1_name column1_datatype [col1_constraints],
        column2_name column2_datatype [col2_constraints],
        ...
        columnN_name columnN_datatype [colN_constraints]

    );

    --Example
    CREATE TABLE business_type (
        id serial PRIMARY KEY,
        description TEXT NOT NULL
    );








