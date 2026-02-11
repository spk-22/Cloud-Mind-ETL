import requests
import pyodbc
import pandas as pd
import time
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ==========================
# 1. SQL CONNECTIONS
# ==========================
CONNECTION_STRINGS = {
    "ETL_DB": "Driver={ODBC Driver 17 for SQL Server};Server=tcp:aietl-sqldb-server-001.database.windows.net,1433;Database=ETL_DB;Uid=CloudSA02c5ee85;Pwd=Spoorthi@1234;Encrypt=YES;TrustServerCertificate=YES;Connection Timeout=300;",
   "ETL_DB_2": (
        "Driver={ODBC Driver 17 for SQL Server};"
        "Server=aietl-sqldb-server-001.database.windows.net,1433;"
        "Database=ETL_DB_2;"
        "UID=CloudSA02c5ee85;"
        "PWD=Spoorthi@1234;"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=2000;"
    ),
    "ETL_DB_3": "Driver={ODBC Driver 17 for SQL Server};Server=tcp:aietl-sqldb-server-001.database.windows.net,1433;Database=ETL_DB_3;Uid=CloudSA02c5ee85;Pwd=Spoorthi@1234;Encrypt=YES;TrustServerCertificate=YES;Connection Timeout=300;"
}

connections = {db: pyodbc.connect(cs) for db, cs in CONNECTION_STRINGS.items()}
logging.info("Connected to all databases ✅")

# ==========================
# 2. SCHEMAS & TABLE NAMES
# ==========================
SCHEMAS = {
    "ETL_DB": ["StockCode","Description","Country","Quantity","UnitPrice","Weekday","Month",
               "Inventory","CompetitorPrice","Quarter","Discount","IsWeekend","PromotionFlag",
               "MarketTrend","Revenue"],
    "ETL_DB_2": ["InvoiceNo","StockCode","Description","Country","Quantity","UnitPrice","Hour","Weekday",
                 "Month","Inventory","CompetitorPrice","TotalDemand"],
    "ETL_DB_3": ["business_segment","lead_type","lead_behaviour_profile","has_company","has_gtin",
                 "average_stock","business_type","avg_customer_age","Quantity","UnitPrice","Hour",
                 "Weekday","Month","Inventory","CompetitorPrice","TotalDemand"]
}

TABLES = {
    "ETL_DB": "StockData",
    "ETL_DB_2": "FACT_DEMAND",
    "ETL_DB_3": "FACT_LEADS"
}

# ==========================
# 3. INSERT FUNCTION (APPEND ONLY)
# ==========================
def insert_dataframe_append(df, table_name, conn, schema):
    cursor = conn.cursor()
    # Keep only schema columns, missing columns filled with None
    df_to_insert = pd.DataFrame({col: df[col] if col in df.columns else None for col in schema})
    
    cols = ",".join(f"[{c}]" for c in schema)
    placeholders = ",".join("?" * len(schema))
    sql = f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})"
    
    rows = df_to_insert.astype(object).where(pd.notnull(df_to_insert), None).values.tolist()
    for row in rows:
        cursor.execute(sql, *row)
    conn.commit()
    logging.info(f"Inserted {len(rows)} rows into {table_name}")

# ==========================
# 4. FETCH DATA FROM FakeStoreAPI
# ==========================
def fetch_products():
    url = "https://fakestoreapi.com/products"
    df = pd.DataFrame(requests.get(url).json())
    # Map API fields to ETL_DB schema
    df_mapped = pd.DataFrame({
        "StockCode": df["id"],
        "Description": df["title"],
        "Country": None,
        "Quantity": np.random.randint(1, 20, size=len(df)),
        "UnitPrice": df["price"],
        "Weekday": None,
        "Month": None,
        "Inventory": np.random.randint(10, 100, size=len(df)),
        "CompetitorPrice": None,
        "Quarter": None,
        "Discount": None,
        "IsWeekend": None,
        "PromotionFlag": None,
        "MarketTrend": None,
        "Revenue": df["price"] * np.random.randint(1,5,len(df))
    })
    return df_mapped

def fetch_carts():
    url = "https://fakestoreapi.com/carts"
    df = pd.DataFrame(requests.get(url).json())
    # Map API fields to ETL_DB_1 schema
    df_mapped = pd.DataFrame({
        "InvoiceNo": df["id"],
        "StockCode": None,
        "Description": None,
        "Country": None,
        "Quantity": None,
        "UnitPrice": None,
        "Hour": None,
        "Weekday": None,
        "Month": None,
        "Inventory": None,
        "CompetitorPrice": None,
        "TotalDemand": None
    })
    return df_mapped

def fetch_users():
    url = "https://fakestoreapi.com/users"
    df = pd.DataFrame(requests.get(url).json())
    # Map API fields to ETL_DB_2 schema
    df_mapped = pd.DataFrame({
        "business_segment": None,
        "lead_type": None,
        "lead_behaviour_profile": None,
        "has_company": df["id"],
        "has_gtin": None,
        "average_stock": np.random.randint(1,100, size=len(df)),
        "business_type": None,
        "avg_customer_age": np.random.randint(18,65, size=len(df)),
        "Quantity": None,
        "UnitPrice": None,
        "Hour": None,
        "Weekday": None,
        "Month": None,
        "Inventory": None,
        "CompetitorPrice": None,
        "TotalDemand": None
    })
    return df_mapped

# ==========================
# 5. ETL LOOP
# ==========================
SLEEP_INTERVAL = 60  # seconds
while True:
    try:
        logging.info("Fetching products...")
        df_products = fetch_products()
        insert_dataframe_append(df_products, TABLES["ETL_DB"], connections["ETL_DB"], SCHEMAS["ETL_DB"])

        logging.info("Fetching carts...")
        df_carts = fetch_carts()
        insert_dataframe_append(df_carts, TABLES["ETL_DB_2"], connections["ETL_DB_2"], SCHEMAS["ETL_DB_2"])

        logging.info("Fetching users...")
        df_users = fetch_users()
        insert_dataframe_append(df_users, TABLES["ETL_DB_3"], connections["ETL_DB_3"], SCHEMAS["ETL_DB_3"])
        logging.info(f"Sleeping {SLEEP_INTERVAL} seconds before next run...\n")
        time.sleep(SLEEP_INTERVAL)
    except Exception as e:
        logging.error(f"ETL failed: {e}. Retrying in 10 seconds...")
        time.sleep(10)
