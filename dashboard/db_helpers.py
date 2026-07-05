import pandas as pd

def run_query(conn, query, params=None):
    """Execute SQL query and return results as pandas DataFrame."""
    return pd.read_sql(query, conn, params=params)
    
def monthly_avg_consumption(conn):
    query = """
        SELECT DATE_FORMAT(timestamp, '%Y-%m') AS month,
               ROUND(AVG(consumption_kwh), 2) AS avg_consumption
        FROM energy_consumption
        GROUP BY month
        ORDER BY month;
    """
    return pd.read_sql(query, conn)

def weekday_vs_weekend(conn):
    query = """
        SELECT CASE WHEN DAYOFWEEK(timestamp) IN (1,7) THEN 'Weekend' ELSE 'Weekday' END AS day_type,
               ROUND(AVG(consumption_kwh), 2) AS avg_consumption
        FROM energy_consumption
        GROUP BY day_type;
    """
    return pd.read_sql(query, conn)

def correlation_temp_consumption(conn):
    query = """
        SELECT 
            (SUM((consumption_kwh - avg_c) * (temperature_c - avg_t)) /
            (COUNT(*) * STDDEV(consumption_kwh) * STDDEV(temperature_c))) AS correlation
        FROM energy_consumption,
             (SELECT AVG(consumption_kwh) AS avg_c, AVG(temperature_c) AS avg_t FROM energy_consumption) AS stats;
    """
    return pd.read_sql(query, conn)

def top_peak_hours(conn):
    query = """
        SELECT HOUR(timestamp) AS hour,
               ROUND(AVG(consumption_kwh), 2) AS avg_consumption
        FROM energy_consumption
        GROUP BY hour
        ORDER BY avg_consumption DESC
        LIMIT 5;
    """
    return pd.read_sql(query, conn)

def consumption_by_house_type(conn):
    query = """
        SELECT h.house_type,
               ROUND(AVG(e.consumption_kwh), 2) AS avg_consumption
        FROM energy_consumption e
        JOIN households h ON e.household_id = h.household_id
        GROUP BY h.house_type;
    """
    return pd.read_sql(query, conn)

def consumption_by_income(conn):
    query = """
        SELECT h.income_level,
               ROUND(AVG(e.consumption_kwh), 2) AS avg_consumption
        FROM energy_consumption e
        JOIN households h ON e.household_id = h.household_id
        GROUP BY h.income_level
        ORDER BY avg_consumption DESC;
    """
    return pd.read_sql(query, conn)

def consumption_by_activity_pattern(conn):
    query = """
        SELECT h.activity_pattern,
               CASE h.activity_pattern
                   WHEN 0 THEN 'Day shift'
                   WHEN 1 THEN 'Night shift'
                   WHEN 2 THEN 'Mixed'
               END AS pattern_label,
               ROUND(AVG(e.consumption_kwh), 2) AS avg_consumption
        FROM energy_consumption e
        JOIN households h ON e.household_id = h.household_id
        GROUP BY h.activity_pattern
        ORDER BY avg_consumption DESC;
    """
    return pd.read_sql(query, conn)

def seasonal_profile(conn):
    query = """
        SELECT MONTH(timestamp) AS month,
               ROUND(AVG(consumption_kwh), 2) AS avg_consumption,
               ROUND(AVG(temperature_c), 2) AS avg_temp
        FROM energy_consumption
        GROUP BY month
        ORDER BY month;
    """
    return pd.read_sql(query, conn)

def daily_profile(conn):
    query = """
        SELECT HOUR(timestamp) AS hour,
               ROUND(AVG(consumption_kwh), 2) AS avg_consumption
        FROM energy_consumption
        GROUP BY hour
        ORDER BY hour;
    """
    return pd.read_sql(query, conn)

def correlation_humidity_consumption(conn):
    query = """
        SELECT 
            (SUM((consumption_kwh - avg_c) * (humidity_percent - avg_h)) /
            (COUNT(*) * STDDEV(consumption_kwh) * STDDEV(humidity_percent))) AS correlation
        FROM energy_consumption,
             (SELECT AVG(consumption_kwh) AS avg_c, AVG(humidity_percent) AS avg_h FROM energy_consumption) AS stats;
    """
    return pd.read_sql(query, conn)

def top_consuming_households(conn, limit=10):
    query = """
        SELECT household_id,
               ROUND(AVG(consumption_kwh), 2) AS avg_consumption,
               COUNT(*) AS records
        FROM energy_consumption
        GROUP BY household_id
        ORDER BY avg_consumption DESC
        LIMIT %s;
    """
    return pd.read_sql(query, conn, params=(limit,))

def consumption_outliers(conn, threshold=2):
    query = """
        SELECT household_id,
               ROUND(AVG(consumption_kwh), 2) AS avg_cons,
               ROUND(STDDEV(consumption_kwh), 2) AS std_cons,
               ROUND(AVG(consumption_kwh) + STDDEV(consumption_kwh) * %s, 2) AS upper_bound
        FROM energy_consumption
        GROUP BY household_id
        HAVING STDDEV(consumption_kwh) > (SELECT AVG(std) FROM (SELECT STDDEV(consumption_kwh) AS std FROM energy_consumption GROUP BY household_id) AS t)
        ORDER BY std_cons DESC
        LIMIT 10;
    """
    return pd.read_sql(query, conn, params=(threshold,))

def get_city_hourly_consumption(conn):
    """Aggregate hourly consumption across all households (city/neighborhood level)."""
    query = """
        SELECT timestamp,
               SUM(consumption_kwh) AS total_consumption,
               AVG(temperature_c) AS temperature_c,
               AVG(humidity_percent) AS humidity_percent
        FROM energy_consumption
        GROUP BY timestamp
        ORDER BY timestamp;
    """
    return pd.read_sql(query, conn)

def get_household_consumption_with_meta(conn):
    query = """
        SELECT e.household_id, e.timestamp, e.consumption_kwh,
               h.house_type, h.income_level, h.activity_pattern
        FROM energy_consumption e
        JOIN households h ON e.household_id = h.household_id;
    """
    return pd.read_sql(query, conn)