-----CREATING POSTGRESQL DATABASES-----

-- CREATING A DATABASE--

    CREATE DATABASE db_name;

-- CREATING TABLES--

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

        CREATE TABLE applicant (
            id serial PRIMARY KEY,
            name TEXT NOT NULL,
            zip_code CHAR(5) NOT NULL,
            business_type_id INTEGER references business_type(id)
            --Defines relationship to other table
        );

-- CREATING SCHEMAS--

    --Similar concept to directory
    --Containing tables and other database objects (datatypes, functions)

    --Uses:
        --Providing database users with seperate environments
            --E.g.  Production Schema
            --      Developer1 Schema
            --      Developer2 Schema
        
        --Organising database objects into related groups
            --E.g. by business unit
            --      Equipment Schema
            --      Energy Schema
            --      Services Schema

    --Default Schema (public)

    CREATE SCHEMA schema_name;

    --E.g.
        CREATE SCHEMA division1;

        CREATE table division1.school (
            id serial PRIMARY KEY,
            name TEXT NOT NULL,
            mascot_name TEXT,
            num_scholarships INTEGER DEFAULT 0
        );

-- INTRODUCTION TO POSTGRESQL DATA TYPES

    -- Data categories
        -- Text
        -- Numeric
        -- Temporal
        -- Boolean
        -- Geometric
        -- Binary
        -- Monetary

    --Examples
        --Representing Dates
            "May 3, 2006" (text)
            "5/3/2006" (text)
            2006-05-03 (date) -- Standardised Postgres Date Format

        --Representing Boolean
            "Yes"/"No" (text)
            "Y"/"N" (text)
            'true'/'false' (boolean) -- Will restrict values to only true/false

        --Representing Distance
            "362 miles" (text)
            "362" (text)
            362 (numeric) -- Advantageous as allows numeric calculations to be made

        -- Example table with datatypes defined
            CREATE TABLE project (
                -- Unique identifier for projects
                id SERIAL PRIMARY KEY,
                -- Whether or not project is franchise opportunity
                is_franchise BOOLEAN DEFAULT FALSE,
                -- Franchise name if project is franchise opportunity
                franchise_name TEXT DEFAULT NULL,
                -- State where project will reside
                project_state TEXT,
                -- County in state where project will reside
                project_county TEXT,
                -- District number where project will reside
                congressional_district NUMERIC,
                -- Amount of jobs projected to be created
                jobs_supported NUMERIC
            );
-- DEFINING TEXT COLUMNS

    CREATE TABLE book (
        isbn CHAR(13),
        author_first_name VARCHAR(50) NOT NULL,
        author_last_name VARCHAR(50) NOT NULL,
        content TEXT NOT NULL
        
    );

    -- TEXT
        -- Strings of variable length
        -- Strings of unlimited length
        -- Good for text based values of unknown length

    -- VARCHAR
        -- Strings of variable length
        -- Strings of unlimited length
        -- Retsrictions can be imposed on column values

        VARCHAR(N)
            -- Maximum number of characters stored N
            -- Can store strings with less than N characters
            -- Inserting longer string is error
            -- VARCHAR with unspecified N equivalent to TEXT
    
    --CHAR
        CHAR(N)
        -- Exactly N characters
        -- Strings right-padded with spaces is less than N
        -- CHAR with unspecified N defaults to CHAR(1)

-- DEFINING NUMERIC DATA COLUMNS

    CREATE TABLE people.employee {
        id SERIAL PRIMARY KEY,
        first_name VARCHAR(10) NOT NULL,
        last_name VARCHAR(10) NOT NULL,
        num_sales INTEGER
        salary DECIMAL(6,2) NOT NULL -- No more than one million
    }

    -- DISCRETE NUMERIC
        SMALLINT -- (-32768, 32767)
        INTEGER -- (-2147483648, 2147483647)
        BIGINT -- (-9223372036854775808, 9223372036854775808)
        SERIAL -- autoincrementing integer 1 to 2147483647

    -- CONTINUOUS NUMERIC
        DECIMAL(precision, scale) -- Up to 131072 digits before decimal, 16383 after
        -- precision : total number of decimal digits, either side of decimal point
        -- scale : number of decimal digits to right of decimal point
        NUMERIC -- Interchangeable with DECIMAL
        REAL -- 6 decimal digits-precision
        DOUBLE PRECISION -- 15 decimal digits precision
-- DEFINING BOOLEAN AND TEMPORAL DATA

    CREATE TABLE book (
        isbn CHAR(13)
        author_first_name VARCHAR(50) NOT NULL,
        author_last_name VARCHAR(50) NOT NULL,
        content TEXT NOT NULL,
        originally_published DATE NOT NULL,
        out_of_print BOOLEAN DEFAULT FALSE
    );

    CREATE TABLE appeal (
        id SERIAL PRIMARY KEY,
        content TEXT NOT NULL,
        received_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        approved_on_appeal BOOLEAN DEFAULT NULL,
        reviewed DATE
    );

    -- BOOLEAN (BOOL interchangeable)
        -- Three possible values
            true
            false
            NULL
        -- Unspecified defaults to false
        -- Can specify default manualy
        in_stock BOOL DEFAULT TRUE;

    -- TEMPORAL
        --TIMESTAMP
            2010-09-21 15:47:16
            CURRENT_TIMESTAMP -- function returning current datetime
        --DATE
            1972-07-08
            CURRENT_DATE -- function returning current date
        --TIME
            05:30:00

-- THE IMPORTANCE OF DATA NORMALISATION

    -- Performed on data tables to protect from data anomalies, ensure integrity of data

    -- Example 1
        -- Data redundancy (duplication) can be problematic

        CREATE TABLE loan (
            borrowed_id INTEGER REFERENCES borrower(id),
            bank_name VARCHAR(50) DEFAULT NONE,
            ...
        );

        CREATE TABLE bank (
            id SERIAL PRIMARY KEY,
            name VARCHAR(50) DEFAULT NULL,
            ...
        );

        -- Problem 1: Different banks/same name
        -- Problem 2: Name changes

        CREATE TABLE loan (
            borrower_id INTEGER REFERENCES borrower(id)
            bank_id INTEGER REFERENCES bank(id)
        )

        -- Banks share name with distinct ids
        -- Updates to bank names will only affect bank table
