-- Feature extraction for the churn model.
--
-- In production the model does not read a CSV. It reads the output of this
-- query, run against the billing warehouse on a nightly schedule. The Python
-- feature module mirrors these transformations so that training and serving
-- stay consistent.
--
-- Dialect: Snowflake / Postgres compatible.

WITH base AS (
    SELECT
        c.customer_id,
        c.gender,
        c.senior_citizen,
        c.partner,
        c.dependents,
        c.tenure_months,
        c.contract_type,
        c.payment_method,
        c.paperless_billing,
        b.monthly_charges,
        b.total_charges
    FROM customers        AS c
    JOIN billing_summary  AS b
      ON b.customer_id = c.customer_id
    WHERE c.is_active_at_snapshot = TRUE
),

services AS (
    -- One row per customer with a flag per subscribed service.
    SELECT
        customer_id,
        MAX(CASE WHEN service_name = 'OnlineSecurity'   AND status = 'active' THEN 1 ELSE 0 END) AS online_security,
        MAX(CASE WHEN service_name = 'OnlineBackup'     AND status = 'active' THEN 1 ELSE 0 END) AS online_backup,
        MAX(CASE WHEN service_name = 'DeviceProtection' AND status = 'active' THEN 1 ELSE 0 END) AS device_protection,
        MAX(CASE WHEN service_name = 'TechSupport'      AND status = 'active' THEN 1 ELSE 0 END) AS tech_support,
        MAX(CASE WHEN service_name = 'StreamingTV'      AND status = 'active' THEN 1 ELSE 0 END) AS streaming_tv,
        MAX(CASE WHEN service_name = 'StreamingMovies'  AND status = 'active' THEN 1 ELSE 0 END) AS streaming_movies,
        MAX(CASE WHEN service_name = 'InternetService'  AND status = 'active' THEN 1 ELSE 0 END) AS has_internet
    FROM customer_services
    GROUP BY customer_id
),

engineered AS (
    SELECT
        b.*,

        COALESCE(s.online_security, 0)
      + COALESCE(s.online_backup, 0)
      + COALESCE(s.device_protection, 0)
      + COALESCE(s.tech_support, 0)
      + COALESCE(s.streaming_tv, 0)
      + COALESCE(s.streaming_movies, 0)                       AS num_addons,

        -- Average spend per month of tenure. Guards against divide-by-zero for
        -- customers in their first billing cycle.
        CASE
            WHEN b.tenure_months > 0 THEN b.total_charges / b.tenure_months
            ELSE b.monthly_charges
        END                                                    AS avg_monthly_spend,

        CASE
            WHEN b.tenure_months = 0 THEN '0-6m'
            WHEN b.tenure_months <= 6 THEN '0-6m'
            WHEN b.tenure_months <= 12 THEN '6-12m'
            WHEN b.tenure_months <= 24 THEN '1-2y'
            WHEN b.tenure_months <= 48 THEN '2-4y'
            ELSE '4y+'
        END                                                    AS tenure_bucket,

        CASE WHEN b.contract_type  = 'Month-to-month'   THEN 1 ELSE 0 END AS is_month_to_month,
        CASE WHEN b.payment_method = 'Electronic check' THEN 1 ELSE 0 END AS is_electronic_check,

        s.has_internet
    FROM base AS b
    LEFT JOIN services AS s
      ON s.customer_id = b.customer_id
)

SELECT
    e.*,
    -- Recent price movement. Positive means the customer is paying more now
    -- than their historic average, which correlates with churn.
    e.monthly_charges - e.avg_monthly_spend AS charge_delta
FROM engineered AS e;
