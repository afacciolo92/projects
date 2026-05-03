USE santir31_db
CREATE SCHEMA customer360;
CREATE TABLE customer360.customer_info (
   customer_id                   INTEGER,
   first_name                    VARCHAR(255),
   last_name                     VARCHAR(255),
   conversion_id                 INTEGER ,
   conversion_number             INTEGER,--(1, 2, 3, etc.)
   conversion_type               VARCHAR(255),--(activation or reactivation)
   conversion_date               DATE,
   conversion_week               VARCHAR(255),
   next_conversion_week          VARCHAR(255),--(week of the immediate next conversion)
   conversion_channel            VARCHAR(255),
   first_order_week              VARCHAR(255),
   first_order_total_paid        NUMERIC,
   week_counter                  INTEGER,--This helps track the sequence of weeks starting from the conversion week. Possible values: 1, 2, 3, etc.
   order_week                    VARCHAR(255),--Represents the specific week of the order, for example 2020-W10, 2020-W11, etc.
   orders_placed                 INTEGER,--Indicates whether orders were placed during an order week (1: orders were placed, 0: no orders placed)
   total_before_discounts        DECIMAL(10,2),--Total of all orders in an order week before discounts
   total_discounts               DECIMAL(10,2),--Total discounts received in the specified order week
   total_paid_in_week            DECIMAL(10,2),--Total paid after discounts in a given order week
   conversion_cumulative_revenue DECIMAL(10,2),--Revenue for the specific conversion up to that week
   lifetime_cumulative_revenue   DECIMAL(10,2)
);



USE mmai_db;
--Join customer id, product name, running week, year week and calculate price before discount on orders table
WITH order_with_product AS (
    SELECT od.*,
           cd.customer_id,
           pd.product_name,
           dd.running_week,
           dd.year_week,
           od.discount_value+od.price_paid AS price_before_discount
    FROM fact_tables.orders AS od
    LEFT JOIN dimensions.product_dimension AS pd
        ON pd.sk_product=od.fk_product
    LEFT JOIN dimensions.date_dimension AS dd
        ON od.fk_order_date=dd.sk_date
    LEFT JOIN dimensions.customer_dimension AS cd
        ON od.fk_customer=cd.sk_customer
), customer_and_conversion AS(
    --Joins customer dimension with conversions table and adds week from date dimensions
    SELECT cd.customer_id,
       cd.first_name,
       cd.last_name,
       cv.conversion_id,
       RANK() OVER (PARTITION BY cd.customer_id ORDER BY cv.conversion_id) AS conversion_number, --Create conversion number
       cv.conversion_type,
       cv.conversion_date,
       dd.year_week AS conversion_week,
       dd.running_week,
       cv.conversion_channel,
       LEAD(dd.year_week) OVER (PARTITION BY cd.customer_id ORDER BY cv.conversion_date) AS next_conversion_week,
       LEAD(dd.running_week) OVER (PARTITION BY cd.customer_id ORDER BY cv.conversion_date) AS next_conversion_run_week,
       owp.year_week AS first_order_week,
       cv.order_number AS first_order_number,
       owp.product_name AS first_order_product,
       owp.price_paid AS first_order_total_paid
    FROM fact_tables.conversions as cv
    INNER JOIN dimensions.customer_dimension AS cd
        ON cd.sk_customer=cv.fk_customer
    LEFT JOIN dimensions.date_dimension AS dd
        ON dd.sk_date=cv.fk_conversion_date
    LEFT JOIN order_with_product AS owp
        ON cv.order_number=owp.order_number
), week_count AS(
    --Counting list with the size of running week from date dimension
    SELECT DISTINCT running_week, year_week
    FROM dimensions.date_dimension AS dd
), conv_with_counter AS (
    --Counter of running week
    SELECT cc.*,
       ROW_NUMBER() over (PARTITION BY cc.customer_id, cc.conversion_number ORDER BY wc.running_week) AS conversion_week_counter,
       cc.running_week+(ROW_NUMBER() over (PARTITION BY cc.customer_id, cc.conversion_number ORDER BY wc.running_week))-1 AS customer_week_counter
    FROM customer_and_conversion AS cc
    INNER JOIN week_count AS wc
        ON wc.running_week>=cc.running_week
), conv_cust_weekCount AS (
    --Add the year week and filter the rows that don't correspond to the correct counter
    SELECT cwc.*, wc.year_week
    FROM conv_with_counter AS cwc
    LEFT JOIN week_count AS wc ON wc.running_week=cwc.customer_week_counter
    WHERE (customer_week_counter<next_conversion_run_week AND next_conversion_run_week IS NOT NULL)
        OR (customer_week_counter<(SELECT MAX(ddim.running_week)+1 FROM dimensions.date_dimension AS ddim) AND next_conversion_run_week IS NULL)
), grouped_orders AS (
    --group orders by week
    SELECT owp.customer_id,
           owp.running_week,
           COUNT(DISTINCT owp.order_number) AS order_placed,
           SUM(price_before_discount) AS total_before_discounts,
           SUM(discount_value) AS total_discounts,
           SUM(price_paid) AS total_paid
    FROM order_with_product AS owp
    GROUP BY owp.customer_id, owp.running_week
)
INSERT INTO santir31_db.customer360.customer_info(customer_id, first_name, last_name, conversion_id, conversion_number, conversion_type, conversion_date, conversion_week, next_conversion_week, conversion_channel, first_order_week, first_order_total_paid, week_counter, order_week, orders_placed, total_before_discounts, total_discounts, total_paid_in_week, conversion_cumulative_revenue, lifetime_cumulative_revenue)

--final query
SELECT ccw.customer_id,
       ccw.first_name,
       ccw.last_name,
       ccw.conversion_id,
       ccw.conversion_number,
       ccw.conversion_type,
       ccw.conversion_date,
       ccw.conversion_week,
       ccw.next_conversion_week,
       ccw.conversion_channel,
       ccw.first_order_week,
      -- ccw.first_order_number,
      -- ccw.first_order_product,
       ccw.first_order_total_paid,
       ccw.conversion_week_counter AS week_counter,
       ccw.year_week AS order_week,
       CASE
           WHEN COALESCE(gor.order_placed, 0)>0 THEN 1
           ELSE 0
        END AS orders_placed,
       COALESCE(gor.total_before_discounts, 0) AS total_before_discounts,
       COALESCE(gor.total_discounts, 0) AS total_discounts,
       COALESCE(gor.total_paid, 0) AS total_paid_in_week,
       SUM(COALESCE(gor.total_paid, 0)) OVER (PARTITION BY ccw.customer_id, ccw.conversion_id ORDER BY ccw.customer_week_counter) AS conversion_cumulative_revenue,
       SUM(COALESCE(gor.total_paid, 0)) OVER (PARTITION BY ccw.customer_id ORDER BY ccw.customer_week_counter) AS lifetime_cumulative_revenue
FROM conv_cust_weekCount AS ccw
LEFT JOIN grouped_orders AS gor
    ON gor.running_week=ccw.customer_week_counter AND gor.customer_id=ccw.customer_id

