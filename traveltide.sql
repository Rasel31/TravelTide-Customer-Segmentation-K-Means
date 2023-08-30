WITH travel_tide AS ( -- Common Table Expression (CTE) named travel_tide
  SELECT 
    * -- retrieves all columns from 
  FROM 
    sessions s -- table sessions and assigns it the alias s.
    LEFT JOIN users u USING (user_id) -- left join between sessions and users tables using the common column user_id.
    LEFT JOIN flights f ON s.trip_id = f.trip_id -- left join between sessions and flights tables based on the common column trip_id.
    LEFT JOIN hotels h ON h.trip_id = s.trip_id -- left join between hotels and sessions tables based on the common column trip_id.
  WHERE 
    s.session_start >= '2023-01-04' -- filters the rows where the session_start date is greater than or equal to '2023-01-04'. 
    AND u.user_id IN ( --  filters the rows where the user_id is found in the subquery result.
      SELECT 
        user_id 
      FROM 
        (
          SELECT 
            user_id, 
            COUNT(*) AS n -- calculates the number of sessions for each user_id since '2023-01-04'.
          FROM 
            sessions 
          WHERE 
            session_start >= '2023-01-04' 
          GROUP BY 
            1
        ) AS session_count 
      WHERE 
        n > 7 -- filters the user_id based on the condition that the session count is greater than 7.
    )
) 

--  retrieves all columns from the CTE travel_tide.
SELECT 
  * 
FROM 
  travel_tide; 


-- 1 How many customers (unique user_id's) are in the customer segmentation cohort data set?

SELECT 
  COUNT(DISTINCT user_id) AS unique_users 
FROM 
  travel_tide;
  
-- Rounded to the nearest integer, how many page_clicks are there on average across all browsing sessions?

Select 
	ROUND(AVG(page_clicks), 0) -- get average page clicks from sessions
FROM 
	sessions;

-- What is the shortest browsing session time resulting in a booking?

SELECT 
	trip_id, session_end - session_start AS booking_time -- get trip id with booking_time
FROM 
	sessions
WHERE 
	trip_id IS NOT NULL -- condition where trip id not empty
ORDER BY 
	booking_time -- order the result
LIMIT
	10;-- showing first 10 rows
	

-- haversine distance calculation for jax, jfk airport


WITH lax AS( -- -- make cte for lax airport lattitide and longitude
  SELECT 
    home_airport_lat as lat1, 
    home_airport_lon AS lng1 
  FROM 
    users 
  WHERE 
    home_airport = 'LAX' 
  LIMIT 
    1
), jfk AS( -- make cte for jfk airport lattitide and longitude
  SELECT 
    home_airport_lat as lat2, 
    home_airport_lon AS lng2 
  FROM 
    users 
  WHERE 
    home_airport = 'JFK' 
  LIMIT 
    1
) 
SELECT 
  ( -- haversine formula
    6371 * acos(
      cos(
        radians(Lat1)
      ) * cos(
        radians(Lat2)
      ) * cos(
        radians(Lng2) - radians(Lng1)
      ) + sin(
        radians(Lat1)
      ) * sin(
        radians(Lat2)
      )
    )
  ) AS distance_km 
FROM 
  lax, 
  jfk









