-----JOINING DATA IN SQL-----

--INTRODUCTION TO JOINS--

    --INNER JOIN
        'Only includes records where the key is in both tables'
        SELECT p1.country, p1.continent,
               prime_minister, president
        FROM prime_ministers AS p1
        INNER JOIN presidents AS p2
        ON p1.country = p2.country;

    --INNER JOIN via USING
        'Where key has same name in both tables we can use USING clause'
        SELECT left_table.id AS L_id,
               left_table.val AS L_val,
               right_table.val AS R_val
        FROM left_table
        INNER JOIN right_table
        USING (id);

        SELECT p1.country, p1.continent,
               prime_minister, president
        FROM prime_ministers AS p1
        INNER JOIN presidents AS p2
        USING (country);

    --SELF JOINS using CASE
        'Joining a table with itself'
        'Used to compare values of a field, to other values of the same field'

        --E.G. Returning pairing of each country, with every other country in the same continent
        SELECT p1.country AS country1
               p2.country AS country2
               p1.continent
        FROM prime_ministers AS p1
        INNER JOIN prime_ministers AS p2
        ON p1.continent = p2.continent
        LIMIT 14;

        --Excluding pairing of country with itself
        SELECT p1.country AS country1
               p2.country AS country2
               p1.continent
        FROM prime_ministers AS p1
        INNER JOIN prime_ministers AS p2
        ON p1.continent = p2.continent AND p1.country <> p2.country
        LIMIT 13;

        --CASE WHEN and THEN
        SELECT name, continent, indep_year,
            CASE WHEN indep_year < 1990 THEN 'before 1990'
                 WHEN indep_year <= 1930 THEN 'between 1990 and 1930'
                 ELSE 'after 1930'
                 AS indep_year_group
        FROM states
        ORDER BY indep_year_group;

--OUTER JOINS AND CROSS JOINS

    --LEFT JOIN
    'Only records where the key is present in the left table'
    SELECT p1.country, prime_minister, president
    FROM prime_ministers AS p1
    LEFT JOIN presidents AS p2
    ON p1.country = p2.country;

    --RIGHT JOIN
    'Only records where the key is present in the left table'
    SELECT p1.country, prime_minister, president
    FROM prime_ministers AS p1
    RIGHT JOIN presidents AS p2
    ON p1.country = p2.country;

    --FULL JOINS
    'Combines a left join and right join'
    SELECT left_table.id AS L_id,
           right_table.id AS R_id,
           left_table.val AS L_val,
           right_table.val AS R_val
    FROM left_table
    FULL JOIN right_table
    USING (id);

    SELECT p1.country AS pm_co, p2.country AS pres_co,
        prime_minister, president
    FROM prime_minister AS p1
    FULL JOIN president AS p2

    --CROSS JOINS
    'Creates all possible combinations of two tables'

    --E.G. Pairing all prime-ministers and presidents in NA, Oceania for meetings
    SELECT prime_minister, president
    FROM prime_ministers AS p1
    CROSS JOIN presidents AS p2
    WHERE p1.continent IN ('North Amercica', 'Oceania')

--SET THEORY CLAUSES

       --UNION
              UNION 'includes all records and does not double count for those in both tables'
              UNION ALL 'includes every record and does replicate those in both columns'

              SELECT prime_minister AS leader, country
              FROM prime_ministers
              UNION
              SELECT monarch, country
              FROM monarchs
              ORDER BY country;

       --INTERSECT
              INTERSECT 'only records found in both tables'

              SELECT id
              FROM left_one
              INTERSECT
              SELECT id
              FROM right_one;

              --E.G. Returns no values as no countries with prime minister the same as the president
              SELECT country, prime_minister AS leader
              FROM prime_ministers
              INTERSECT
              SELECT country, president
              FROM presidents;

       --EXCEPT
              EXCEPT 'records only found in one table and not other'
              
              --E.G. returning monarchs that aren't also prime minister
              SELECT monarch, country
              FROM monarchs
              EXCEPT
              SELECT prime_minister, country
              FROM prime_ministers;

       --SEMI-JOINS and ANTI-JOINS
              'Use right table to determine which records to keep in the left table'
              
              --SEMI-JOIN
              'Chooses records in the first table where a condition is met in the second table'
              SELECT president, country, continent
              FROM presidents
              WHERE country IN
                     (SELECT name
                      FROM states
                      WHERE indep_year < 1800);

              --ANTI-JOIN
              'Chooses records in the first table where a condition is not met in the second table'
              SELECT president, country, continent
              FROM presidents
              WHERE continent LIKE '%America'
                     AND country NOT IN
                            (SELECT name
                             FROM states
                             WHERE indep_year < 1800);

--SUBQUERIES

       --SUBQUERIES inside WHERE and SELECT clauses

              --E.G. Subquery in WHERE clause
             SELECT name, fert_rate
             FROM states
             WHERE continent = 'Asia'
                     AND  fert_rate < (SELECT AVG(fert_rate)
                                       FROM states);
              --E.G. Subquery in SELECT clause
              SELECT  DISTINCT continent,
                     (SELECT COUNT(*)
                      FROM states
                      WHERE prime_ministers.continent = states.continent) AS countries_num
              FROM prime_ministers;

       --SUBQUERIES inside FROM clauses

              --E.G. Subquery in FROM clause
              SELECT DISTINCT monarchs.continent, subquery.max_perc
              FROM monarchs,
                     (SELECT continent, MAX(women_parli_perc) AS max_perc
                      FROM states
                      GROUP BY continent) AS subquery
              WHERE monarchs.continent = subquery.continent
              ORDER BY continent;
               







              



