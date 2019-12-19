-----INTRODUCTION TO SQL-----

--SELECTING DATA

    --Select Columns
    SELECT name, birthday FROM people;

    --Select All
    SELECT * FROM people;

    --Limited number of rows
    SELECT * FROM people LIMIT 10;

    --Select unique values
    SELECT DISTINCT country FROM films;

    --Count records
    SELECT COUNT(*) FROM reviews

    --Count non-missing values in specific column
    SELECT COUNT(birthdate) FROM people;

    --Count number of distinct values in column
    SELECT COUNT(DISTINCT birthdate) FROM people;



--FILTERING RESULTS

    --WHERE
        SELECT title FROM films WHERE title = 'Metropolis';
        SELECT title FROM films WHERE release_year > 2000;

    --AND
        SELECT title, release_year
        FROM films
        WHERE release_year < 2000 AND
        language = 'Spanish';
    
    --OR
        SELECT title
        FROM films
        WHERE release_year = 1994
        OR release_year = 2000;

    --AND OR
        SELECT title
        FROM films
        WHERE (release_year = 1994 OR release_year = 1995)
        AND (certification = 'PG' OR certification = 'R')

    --BETWEEN
        SELECT title, release_year FROM films
        WHERE release_year BETWEEN 1990 AND 2000
        AND budget > 100*10^6
        AND (language = 'Spanish' OR language = 'French');

    --IN
        SELECT title, release_year FROM films
        WHERE release_year IN (1990, 2000)
        AND duration > 120;

        SELECT title, language FROM films
        WHERE language IN ('English', 'Spanish','French');

    --NULL / IS NULL
        SELECT COUNT(*)
        FROM people
        WHERE birthdate IS NULL;

        SELECT name FROM people
        WHERE birthdate IS NOT NULL;

    --LIKE / NOT LIKE
        'Used to match with wildcards'
        '% will match zero, one or many characters in text'
        '_ will match a single character'
        
        --All names starting with B
        SELECT name FROM people
        WHERE name LIKE 'B%'; 

        --All names with r as second letter
        SELECT name FROM people
        WHERE name LIKE '_r%';

        --All names not starting with A
        SELECT name FROM people
        WHERE name NOT LIKE 'A%'

--AGGREGATE FUNCTIONS
    
    --Aggregate Functions
        SELECT AVG(budget) FROM films
        SELECT MAX(budget) FROM films
        SELECT MIN(budget) FROM films
        SELECT SUM(budget) FROM films

    --Combining agg functions with WHERE
        SELECT SUM(gross) FROM films
        WHERE release_year >= 2000;

        SELECT AVG(gross) FROM films
        WHERE title LIKE 'A%'

        SELECT MAX(gross) FROM films
        WHERE release_year
        BETWEEN 2000 AND 2012;

    --Arithmetic

        --Multiplication
        SELECT (4*3) --Returns 12

        --Division
        'INT/INT returns INT'
        SELECT (4/3) --Returns 1
        'FLOAT/FLOAT returns FLOAT'
        SELECT (4.0/3.0) --Returns 1.333

    --Aliasing

        SELECT MAX(budget) AS max_budget,
               MAX(duration) AS max_duration
        FROM films;

        SELECT title, gross - budget AS net_profit
        FROM films;

        SELECT title, duration / 60.0 AS duration_hours
        FROM films;

        SELECT AVG(duration)/60.0 AS avg_duration_hours
        FROM films;

        SELECT 100.0*COUNT(deathdate)/COUNT(*)
        AS percentage_dead
        FROM people;

        SELECT MAX(release_year) - 
               MIN(release_year)
        AS difference
        FROM films;

--SORTING, GROUPING AND JOINS

    --ORDER BY (Default- ASC)

        SELECT name
        FROM people
        ORDER BY name;

        SELECT title
        FROM films
        ORDER BY release_year DESC;

        SELECT title, release_year
        FROM films
        WHERE release_year IN (2000,2012)
        ORDER BY release_year;

        SELECT *
        FROM films
        WHERE release_year NOT IN (2015)
        ORDER BY duration;

        SELECT title, gross
        FROM films
        WHERE title LIKE 'M%'
        ORDER BY title;

    --Sorting multiple columns
        SELECT birthdate, name
        FROM people
        ORDER BY birthdate, name;

    --GROUP BY
        SELECT sex, count(*)
        FROM employees
        GROUP BY sex
        ORDER BY count DESC;

        SELECT release_year, MAX(budget)
        FROM films
        GROUP BY release_year;

        SELECT language, SUM(gross)
        FROM films
        GROUP BY language;

        SELECT release_year, country, MAX(budget)
        FROM films
        GROUP BY country, release_year
        ORDER BY release_year, country;
    
    --HAVING
        'Filtering result of aggregate functions'

        SELECT release_year
        FROM films
        GROUP BY release_year
        WHERE COUNT(title) > 10;

        SELECT release_year,
        AVG(budget) AS avg_budget,
        AVG(gross) AS avg_gross
        FROM films
        WHERE release_year > 1990
        GROUP BY release_year
        HAVING AVG(budget) > 60*10^6
        ORDER BY avg_gross DESC;

        -- select country, average budget, average gross
        SELECT country,
        AVG(budget) AS avg_budget,
        AVG(gross) AS avg_gross
        -- from the films table
        FROM films
        -- group by country 
        GROUP BY country
        -- where the country has more than 10 titles
        HAVING COUNT(title) > 10
        -- order by country
        ORDER BY country
        -- limit to only show 5 results
        LIMIT 5;



        



    