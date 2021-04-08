
--SELECTING AND FILTERING
SELECT * FROM public.test
WHERE email = 'test@gmail.com'

--% WILDCARD
SELECT email FROM public.test
WHERE email LIKE '%@gmail.com'

SELECT email FROM public.test
WHERE email LIKE '%gmail%'

--OR STATEMENT
SELECT * FROM public.customers
WHERE email LIKE '%@smith%' OR id = 4

--NOT STATEMENT
SELECT * FROM public.customers
WHERE NOT email LIKE '%@smith%'

--ORDERING DATA
--ascending default, if not specified
SELECT * FROM public.customers
ORDER BY name DESC

SELECT * FROM public.customers
ORDER BY name ASC

