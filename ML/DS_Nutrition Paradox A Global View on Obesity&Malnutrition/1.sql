-- In MySQL CLI or Workbench
CREATE DATABASE IF NOT EXISTS nutrition_db;
USE nutrition_db;
select* from df_obesity;
select*from df_malnutrition;

-- 1.Top 5 regions with the highest average obesity levels in the most recent year(2022)
select region,year, avg(Mean_Estimate) as avg_obesity_level from df_obesity
where year = 2022
group by Region order by avg_obesity_level desc limit 5;

-- 2.Top 5 countries with highest obesity estimates
SELECT year, Mean_Estimate FROM df_obesity
WHERE Country = 'India' AND Gender = 'Both' 
ORDER BY year ASC;

-- 3. Obesity trend in India over the years (Mean_estimate)
select year,Gender,Mean_Estimate from df_obesity
where Country = 'india'
order by year asc;

-- 4. Average obesity by gender (across all years and regions)
SELECT Gender, AVG(Mean_Estimate) AS Average_Obesity
FROM df_obesity
GROUP BY Gender;

-- 5. Country count by obesity level category and age group
select obesity_level,age_group ,count(distinct Country) as country_count
from df_obesity group by obesity_level,age_group
order by obesity_level,age_group;

-- 6. Combined Least Reliable & Most Consistent Countries by CI_Width
(
    SELECT 
        Country,
        AVG(CI_Width) AS Avg_CI_Width,
        'Least Reliable' AS Reliability_Type
    FROM df_obesity
    GROUP BY Country
    ORDER BY Avg_CI_Width DESC
    LIMIT 5
)
UNION ALL
(
    SELECT 
        Country,
        AVG(CI_Width) AS Avg_CI_Width,
        'Most Consistent' AS Reliability_Type
    FROM df_obesity
    GROUP BY Country
    ORDER BY Avg_CI_Width ASC
    LIMIT 5
);

-- 7. Average obesity by age group (across all years)
SELECT age_group,AVG(Mean_Estimate) AS Average_Obesity
FROM df_obesity
GROUP BY age_group;

-- 8. Top 10 Countries with consistent low obesity (low average + low CI) over the years
SELECT Country,AVG(Mean_Estimate) AS Average_Obesity,AVG(CI_Width) AS Average_CI_Width
FROM df_obesity
GROUP BY Country
ORDER BY Average_Obesity ASC, 
Average_CI_Width ASC 
LIMIT 10;

-- 9. Countries where female obesity exceeds male by large margin (same year)
WITH GenderData AS (
    SELECT
        year,
        Country,
        MAX(CASE WHEN Gender = 'Female' THEN Mean_Estimate END) AS Female_Estimate,
        MAX(CASE WHEN Gender = 'Male' THEN Mean_Estimate END) AS Male_Estimate
    FROM
        df_obesity
    WHERE
        Gender IN ('Female', 'Male')
    GROUP BY
        year, Country
)
SELECT
    year,
    Country,
    Female_Estimate,
    Male_Estimate,
    (Female_Estimate - Male_Estimate) AS Female_Margin
FROM GenderData
WHERE (Female_Estimate - Male_Estimate) > 5 
ORDER BY Female_Margin DESC;

-- 10. Global average obesity percentage per year
SELECT year,
AVG(Mean_Estimate) AS Global_Average_Obesity
FROM df_obesity
WHERE Gender = 'Both' 
GROUP BY year
ORDER BY year ASC;

-- MalnutritionData

-- 1. Average malnutrition by age group

SELECT Age_Group,AVG(Mean_Estimate) AS Average_Malnutrition
FROM df_malnutrition
GROUP BY Age_Group;

-- 2. Top 5 countries with highest malnutrition (Mean_Estimate)
SELECT Country,
Mean_Estimate FROM df_malnutrition
ORDER BY Mean_Estimate DESC
LIMIT 5;

-- 3. Malnutrition trend in African region over the years
SELECT Year,
AVG(Mean_Estimate) AS Average_Malnutrition_Africa
FROM df_malnutrition
WHERE Region = 'Africa' AND Gender = 'Both' 
GROUP BY Year
ORDER BY Year ASC;

-- 4. Gender-based average malnutrition
SELECT Gender,AVG(Mean_Estimate) AS Average_Malnutrition
FROM df_malnutrition
GROUP BY Gender;
    
-- 5. Malnutrition level-wise (average CI_Width by age group)
SELECT Malnutrition_Level,
    Age_Group,
    AVG(CI_Width) AS Average_CI_Width
FROM
    df_malnutrition
GROUP BY
    Malnutrition_Level,
    Age_Group
ORDER BY
    Malnutrition_Level,
    Age_Group;
    
-- 6. Yearly malnutrition change in specific countries (India, Nigeria, Brazil)
SELECT Year,Country,Mean_Estimate
FROM df_malnutrition
WHERE Country IN ('India', 'Nigeria', 'Brazil')AND Gender = 'Both'
ORDER BY Country,
    Year ASC;
    
    -- 7. Regions with lowest malnutrition averages
    SELECT
    Region,
    AVG(Mean_Estimate) AS Average_Malnutrition
FROM
    df_malnutrition
GROUP BY
    Region
ORDER BY
    Average_Malnutrition ASC
LIMIT 5; 
-- 8. Countries with increasing malnutrition
SELECT
    Country,
    MIN(Mean_Estimate) AS Earliest_Malnutrition,
    MAX(Mean_Estimate) AS Latest_Malnutrition,
    (MAX(Mean_Estimate) - MIN(Mean_Estimate)) AS Malnutrition_Increase
FROM
    df_malnutrition
GROUP BY
    Country
HAVING
    Malnutrition_Increase > 0 
ORDER BY
    Malnutrition_Increase DESC;
    
-- 9. Min/Max malnutrition levels year-wise comparison
SELECT Year,
MIN(Mean_Estimate) AS Minimum_Country_Malnutrition,
MAX(Mean_Estimate) AS Maximum_Country_Malnutrition
FROM df_malnutrition
GROUP BY Year
ORDER BY Year ASC;

-- 10. High CI_Width flags for monitoring (CI_width > 5)
SELECT Year,Country,Gender,Mean_Estimate,CI_Width
FROM df_malnutrition
WHERE CI_Width > 5
ORDER BY CI_Width DESC,
Year DESC;

-- Combined SQL Queries for Obesity and Malnutrition
SELECT O.Country,
AVG(O.Mean_Estimate) AS Avg_Obesity,
AVG(M.Mean_Estimate) AS Avg_Malnutrition
FROM df_obesity O
INNER JOIN df_malnutrition M ON O.Country = M.Country
WHERE O.Country IN ('India', 'USA', 'Nigeria', 'Brazil', 'France') 
GROUP BY O.Country
ORDER BY O.Country;

-- 2. Gender-based Disparity in Both Obesity and Malnutrition
SELECT
    O.Gender, 
    AVG(O.Mean_Estimate) AS Avg_Obesity_Estimate,
    AVG(M.Mean_Estimate) AS Avg_Malnutrition_Estimate,
    (AVG(O.Mean_Estimate) - AVG(M.Mean_Estimate)) AS Disparity_Obesity_Malnutrition
FROM
    df_obesity O
INNER JOIN
    df_malnutrition M 
    ON O.Year = M.Year 
    AND O.Gender = M.Gender 
    AND O.Country = M.Country
WHERE
    O.Gender IN ('Male', 'Female') 
GROUP BY
    O.Gender; 
    
-- 3. Average Obesity and Malnutrition Side-by-Side (Overall)
SELECT
    O.Country,
    AVG(O.Mean_Estimate) AS Avg_Obesity_Estimate,
    AVG(M.Mean_Estimate) AS Avg_Malnutrition_Estimate,
    (AVG(O.Mean_Estimate) - AVG(M.Mean_Estimate)) AS Difference_Obesity_Minus_Malnutrition
FROM
    df_obesity O
INNER JOIN
    df_malnutrition M
    ON O.Country = M.Country AND O.Year = M.Year
GROUP BY
    O.Country
ORDER BY
    O.Country;

-- 4. Annual Global Averages for Both Conditions
SELECT
    O.Year,
    AVG(O.Mean_Estimate) AS Global_Avg_Obesity,
    AVG(M.Mean_Estimate) AS Global_Avg_Malnutrition
FROM
    df_obesity O
INNER JOIN
    df_malnutrition M
    ON O.Year = M.Year
    AND O.Gender = M.Gender
    AND O.Country = M.Country
WHERE
    O.Gender = 'Both' 
GROUP BY
    O.Year
ORDER BY
    O.Year ASC;

-- 4. Comparing Data Reliability (CI_Width) by Age Group
SELECT
    O.Age_Group,
    AVG(O.CI_Width) AS Avg_Obesity_CI_Width,
    AVG(M.CI_Width) AS Avg_Malnutrition_CI_Width,
    (AVG(O.CI_Width) - AVG(M.CI_Width)) AS CI_Width_Difference
FROM
    df_obesity O
INNER JOIN
    df_malnutrition M
    ON O.Age_Group = M.Age_Group
    AND O.Year = M.Year
    AND O.Country = M.Country
    AND O.Gender = M.Gender
GROUP BY
    O.Age_Group
ORDER BY
    O.Age_Group;


