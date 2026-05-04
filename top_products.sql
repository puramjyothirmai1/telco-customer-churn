-- Worthwhile Remote Data Scientist Assessment - Part Two, Question 1
-- Author: Jyothirmai Puram
--
-- Task: Find the top 5 products by total sales from a table named `sales`
--       with columns: product_id, sale_date, amount.
--
-- Notes:
--   * `LIMIT 5` works on PostgreSQL, MySQL, SQLite, Snowflake, Redshift, BigQuery.
--   * On SQL Server use `SELECT TOP 5` instead of `LIMIT 5`.
--   * Add a WHERE clause on `sale_date` if the business cares about a window
--     (e.g. last quarter, current fiscal year).

SELECT
    product_id,
    SUM(amount) AS total_sales
FROM sales
GROUP BY product_id
ORDER BY total_sales DESC
LIMIT 5;
