# Intermediate SQL

- [Intermediate SQL](#intermediate-sql)
  - [CASE statements](#case-statements)
    - [Simple CASE statements](#simple-case-statements)
    - [More complex logic](#more-complex-logic)
    - [Filter out NULLs](#filter-out-nulls)
    - [CASE WHEN with aggregate functions](#case-when-with-aggregate-functions)
  - [Subqueries](#subqueries)
    - [What is a subquery?](#what-is-a-subquery)
    - [Simple subqueries](#simple-subqueries)
    - [Subqueries in FROM](#subqueries-in-from)
    - [Subqueries in SELECT](#subqueries-in-select)
    - [Multiple subqueries](#multiple-subqueries)
  - [Correlated Queries, Nested Queries and CTEs](#correlated-queries-nested-queries-and-ctes)
    - [Correlated Subqueries](#correlated-subqueries)
    - [Nested Subqueries](#nested-subqueries)
    - [Common Table Expressions](#common-table-expressions)
  - [Window Functions](#window-functions)
    - [Introduction to Window Functions](#introduction-to-window-functions)
    - [Window Partitions](#window-partitions)
    - [Sliding Windows](#sliding-windows)

## CASE statements

### Simple CASE statements
 
 ```sql
 SELECT
    id,
    home_goal,
    away_goal,
    CASE WHEN home_goal > away_goal THEN 'Home Team Win'
         WHEN home_goal < away_goal THEN 'Away Team Win'
         ELSE 'Tie' END AS outcome
FROM match
WHERE season = '2013/2014';
 ```

### More complex logic
```sql
SELECT date, hometeam_id, awayteam_id,
   CASE WHEN hometeam_id = 8455 AND home_goal > away_goal
         THEN 'Chelsea home win!'
      WHEN awayteam_id = 8455 AND home_goal < away_goal
         THEN ' Chelsea away win!'
      ELSE 'Loss or tie :)' END AS outcome
FROM match
WHERE hometeam_id = 8455 OR awayteam_id = 8455;
```

### Filter out NULLs

If we do not specify an `ELSE` clause, then any rows where the conditions set are not met will yield `NULL`. We can filter these by using a `WHERE` clause to select only rows where the result is `NOT NULL`.

```sql
SELECT date, season,
   CASE WHEN hometeam_id = 8455 AND home_goal > away_goal
         THEN 'Chelsea home win!'
      WHEN awayteam_id = 8455 AND home_goal < away_goal
         THEN ' Chelsea away win!'
   END AS outcome
FROM match
WHERE CASE WHEN hometeam_id = 8455 AND home_goal > away_goal
         THEN 'Chelsea home win!'
      WHEN awayteam_id = 8455 AND home_goal < away_goal
         THEN ' Chelsea away win!'
      END IS NOT NULL;
```

### CASE WHEN with aggregate functions

`CASE WHEN` with `COUNT`:

```sql
SELECT
   season,
   COUNT(CASE WHEN hometeam_id = 8650
            AND home_goal > away_goal
            THEN id END) AS home_wins,
   COUNT(CASE WHEN awayteam_id = 8650
            AND away_goal > home_goal
            THEN id END) AS away_wins
FROM match
GROUP BY season;
```

`CASE WHEN` with `SUM`:

```sql
SELECT
   season,
   SUM(CASE WHEN hometeam_id = 8650
         THEN home_goal END) AS home_goals,
   SUM(CASE WHEN awayteam_id = 8650
         THEN away_goal END) AS away_goals
FROM match
GROUP BY season;
```

`CASE WHEN` with `AVERAGE`:

```sql
SELECT
   season,
   ROUND(AVG(CASE WHEN hometeam_id = 8650
         THEN home_goal END),2) AS home_goals,
   ROUND(AVG(CASE WHEN awayteam_id = 8650
         THEN away_goal END),2) AS away_goals
FROM match
GROUP BY season;
```

Percentages with `CASE` and `AVG`:

```sql
SELECT
   season,
   AVG(CASE WHEN hometeam_id = 8455 AND home_goal > away_goal THEN 1
            WHEN hometeam_id = 8455 AND home_goal < away_goal THEN 0
      END) AS pct_homewins,
   AVG(CASE WHEN awayteam_id = 8455 AND away_goal > home_goal THEN 1
            WHEN awayteam_id = 8455 AND away_goal < home_goal THEN 0
      END) AS pct_awaywins
FROM match
GROUP BY season;
```

## Subqueries

### What is a subquery?

- A query nested inside another query.
- Useful for intermediary transformation.
- Can be in any part of query: `SELECT`, `FROM`, `WHERE`, `GROUP BY`

```sql
SELECT column
FROM (SELECT column
      FROM table) AS subquery;
```

### Simple subqueries

```sql
SELECT home_goal
FROM match
WHERE home_goal > (
   SELECT AVG(home_goal)
   FROM match);
```

```sql
SELECT
   team_long_name,
   team_short_name AS abr
FROM team
WHERE
   team_api_id IN
   (SELECT hometeam_id
   FROM match
   WHERE country_id = 15722)
```

### Subqueries in FROM

- Restructure and transform your data
  - Transforming data from long to wide before selecting
  - Prefiltering data
  - Calculating aggregates of aggregates

```sql
SELECT team, home_avg
FROM (SELECT
         t.team_long_name AS team,
         AVG(m.home_goal) AS home_avg
      FROM match AS m
      LEFT JOIN team AS t
      ON m.hometeam_id = t.team_api_id
      WHERE season = '2011/2012'
      GROUP BY team) AS subquery
ORDER BY home_avg DESC,
LIMIT 3;
```

### Subqueries in SELECT

- Returns single value
  - Including aggregare values to compare to individual ones.
  - Mathematical calculations

```sql
SELECT
   season,
   COUNT(id) AS matches,
   (SELECT COUNT(id) FROM match) as total_matches
FROM match
GROUP BY season;
```

```sql
SELECT
   date
   (home_goal + away_goal) AS goals,
   (home_goal + away_goal) -
      (SELECT AVG(home_goal + away_goal)
      FROM match
      WHERE season = '2011/2012') AS diff
FROM match
WHERE season = '2011/2012';
```

### Multiple subqueries

```sql
SELECT
   country_id,
   ROUND(AVG(matches.home_goal + matches.away_goal),2) AS avg_goals,
   (SELECT ROUND(AVG(home_goal + away_goal),2)
   FROM match WHERE season = '2013/2014') AS overall_avg
FROM (SELECT
         id,
         home_goal,
         away_goal,
         season
      FROM match
      WHERE home_goal > 5) AS matches
WHERE matches.season = '2013/2014'
   AND (AVG(matches.home_goal + matches.away_goal) >
       (SELECT AVG(home_goal + away_goal)
       FROM match WHERE season = '2013/2014')
GROUP BY country_id;
```

## Correlated Queries, Nested Queries and CTEs

### Correlated Subqueries

- Uses values from the outer query to generate a result
- Re-run for every row generated in the final data set
- Used for advanced joining, filtering and evaluating data

```sql
-- Which match stages tend to have higher than average number of goals scored

-- Simple Query
SELECT
   s.stage,
   ROUND(s.avg_goal,2) AS avg_goal,
   (SELECT AVG(home_goal + away_goal)
   FROM match
   WHERE season = '2012/2013') AS overall_avg
FROM (SELECT
         stage,
         AVG(home_goal + away_goal) AS avg_goals
      FROM match
      WHERE season = '2012/2013'
      GROUP BY stage) AS s
WHERE s.avg_goals > (SELECT AVG(home_goal + away_goal)
                     FROM match
                     WHERE season = '2012/2013');

-- Correlated Query
SELECT
   s.stage,
   ROUND(s.avg_goals,2) AS avg_goal,
   (SELECT AVG(home_goal + away+goal)
   FROM match
   WHERE season = '2012/2013') AS overall_avg
FROM
   (SELECT
      stage,
      AVG(home_goal + away_goal) AS  avg_goals
   FROM match
   WHERE season = '2012/2013'
   GROUP BY stage) AS s
WHERE s.avg_goals > (SELECT AVG(home_goal + away_goal)
                     FROM match AS m
                     WHERE s.stage > m.stage);

```

Differences:

| Simple Subquery                              | Correlated                                            |
| --------------------------------             | --------------------------------                      |
| Can be run independently from the main query | Dependent on the main query to execute                |
| Evaluated once in the whole query            | Evaluated in loops - significantly slows down runtime |

```sql
-- What is the average number of goals scored in each country

-- Simple Subquery
SELECT
   c.name AS country,
   AVG(m.home_goal + m.away_goal)
      AS avg_goals
FROM country AS c
LEFT JOIN match AS m
ON c.id = m.country_id
GROUP BY country;


-- Correlated Subquery
SELECT
   c.name AS country,
   (SELECT
      AVG(home_goal + away_goal)
    FROM match AS m
    WHERE m.country_id = c.id)
      AS avg_goals
FROM country AS c
GROUP BY country;
```

### Nested Subqueries

```sql

-- How much did each countries average differ from the overall average?

--Simple Subquery
SELECT
   c.name AS country,
   AVG(m.home_goal + m.away_goal) AS avg_goals,
   AVG(m.home_goal + m.away_goal) -
      (SELECT AVG(home_goal + away_goal)
       FROM match) AS avg_diff
FROM country AS c
LEFT JOIN match AS m
ON c.id = m.country_id
GROUP BY country;
```

```sql
-- How does each month's total goals differ from the average monthly total of goals scored?

--Nested Subquery
SELECT
   EXTRACT(MONTH FROM date) AS month,
   SUM(m.home_goal + m.away_goal) AS total_goals,
   SUM(m.home_goals + m.away_goals) -
      (SELECT AVG(goals)
       FROM (SELECT
               EXTRACT(MONTH FROM date) AS month,
               SUM(home_goal + away_goal) AS goals
             FROM match
             GROUP BY month)) AS avg_diff
   FROM match AS m
   GROUP BY month;
```

- Can be correlated or uncorrelated, or a combination of the two.
- Can reference information from the outer subquery or main query.

```sql
-- What is each country's average goals scored in the 2011/2012 season?

-- Correlated Nested Subquery
SELECT
   c.name AS country,
   (SELECT AVG(home_goal + away_goal)
    FROM match AS m
    WHERE m.country_id = c.id
          AND id IN (
               SELECT id
               FROM match
               WHERE season = '2011/2012')) AS avg_goals
FROM country AS c
GROUP BY country;
```

### Common Table Expressions

When adding subqueries, complexity increases quickly. CTE's can help us with this.

```sql
WITH cte AS (
   SELECT col1, col2
   FROM table)
SELECT
   AVG(col1) AS avg_col
FROM cte;
```

```sql
-- Subquery in FROM
SELECT
   c.name AS country,
   COUNT(s.id) AS matches
FROM country AS c
INNER JOIN (
   SELECT country_id, id
   FROM match
   WHERE (home_goal + away_goal) >= 10) AS s
ON c.id = s.country_id
GROUP BY country;

-- With CTE

WITH s AS (
   SELECT country_id, id
   FROM match
   WHERE (home_goal + away_goal) >= 10
)

SELECT c.name AS country,
   c.name AS country,
   COUNT(s.id) AS matches
FROM country AS c
INNER JOIN s
ON c.id = s.country_id
GROUP BY country;
```

Multiple CTEs:
- Seperate with comma in `WITH` statement

```sql
WITH s1 AS (
   SELECT country_id, id
   FROM match
   WHERE (home_goal + away_goal) >= 10),
s2 AS (
   SELECT country_id, id
   FROM match
   WHERE (home_goal + away_goal) <=1
)
SELECT
   c.name AS country,
   COUNT(s1.id) AS high_scores,
   COUNT(s2.id) AS low_scores,
FROM country AS c
INNER JOIN s1
ON c.id = s1.country_id
INNER JOIN s2
ON c.id = s2.country_id
GROUP BY country;
```

Why use CTEs?
- Executed only once and stored in memory, leading to improved query performance.
- Improves organisation of queries.
- Can reference itself in a "recursive CTE" (`SELF JOIN`).

## Window Functions

### Introduction to Window Functions

- Performs calculations on an already generated result set (a window).
- Aggregate calculations, without having to group data
  - Similar to subqueries in SELECT
  - Running totals, rankings, moving averages

```sql
-- How many goals were scored in each match in 2011/2012, and how did that compare to the average?

-- Using a subquery
SELECT
   date,
   (home_goal + away_goal) AS goals,
   (SELECT AVG(home_goal + away_goal)
    FROM match
    WHERE season = '2011/2012') AS overall_avg
FROM match
WHERE season = '2011/2012';

-- Using a window function
SELECT
   date,
   (home_goal + away_goal) AS goals,
   AVG(home_goal + away_goal) OVER() as overall_avg
FROM match
WHERE season = '2011/2012';
```

```sql
--What is the rank of matches based on the number of goals scored?

SELECT
   date,
   (home_goal + away_goal) AS goals,
   RANK() OVER(ORDER BY home_goal + away_goal DESC) AS goals_rank
FROM match
WHERE season = '2011/2012';
```

Key differences:
- Processed after every part of a query, except `ORDER BY`
  - Uses information in result set rather than database

- Not available in SQLite

### Window Partitions

- Calculate seperate values for different categories
- Calculate different calculations in the same column

```sql
-- How many goals were scored in each match, and how did that compare to the season's average?

SELECT
   date,
   (home_goal + away_goal) AS goals,
   AVG(home_goal + away_goal)
      OVER(PARTITION BY season) AS season_avg
FROM match;
```

Partition by Multiple Columns:

```sql
SELECT
   c.name,
   m.season,
   (home_goal + away_goal) AS goals,
   AVG(home_goal + away_goal)
      OVER(PARTITION BY m.season, c.name) AS season_ctry_avg
FROM country AS c
LEFT JOIN match AS m
ON c.id = m.country_id
```

Partition considerations:
- Can partition data by 1 or more columns
- Can partition aggregate calculations, ranks etc.

### Sliding Windows

- Perform calculations relative to the current row
- Can be used to calculate running totals, sums, averages etc.
- Can be partitioned by one or more columns

```sql
ROWS BETWEEN <start> AND <finish>

-- Can specify
PRECEDING
FOLLOWING
UNBOUNDED PRECEDING
UNBOUNDED FOLLOWING
CURRENT ROW
```

```sql
SELECT
   date,
   home_goal,
   away_goal,
   SUM(home_goal)
      OVER(ORDER BY date ROWS BETWEEN
           UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total
FROM match
WHERE hometeam_id = 8456 AND season = '2011/2012';
```

```sql
SELECT
   date,
   home_goal
   away_goal,
   SUM(home_goal)
      OVER(ORDER BY date
      ROWS BETWEEN 1 PRECEDING
      AND CURRENT ROW) AS last2
FROM match
WHERE hometeam_id = 8456
      AND season = '2011/2012';


