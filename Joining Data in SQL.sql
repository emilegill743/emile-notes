-----JOINING DATA IN SQL-----

--INTRODUCTION TO JOINS--

    --INNER JOIN
        'Only includes records where the key is in both tables'
        SELECT p1.country, p1.continent,
               prime_minister, president
        FROM prime_ministers AS p1
        INNER JOIN presidents AS p2
        ON p1.country = p2.country;

    --LEFT JOIN
        ''