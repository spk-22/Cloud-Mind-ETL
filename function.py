from azure.storage.blob import BlobServiceClient
import pyodbc
import pandas as pd
import numpy as np
from io import BytesIO

# =====================================================
# 1. BLOB STORAGE CONFIG
# =====================================================


CONTAINER_NAME = "e-commerce"

blob_service = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
container_client = blob_service.get_container_client(CONTAINER_NAME)

print("Blob Storage Connected ✅")

# =====================================================
# 2. SCHEMA DEFINITIONS
# =====================================================
ETL_DB_SCHEMA = {
    "StockCode", "Description", "Country", "Quantity", "UnitPrice",
    "Weekday", "Month", "Inventory", "CompetitorPrice",
    "Quarter", "Discount", "IsWeekend", "PromotionFlag",
    "MarketTrend", "Revenue"
}

ETL_DB1_SCHEMA = {
    "InvoiceNo", "StockCode", "Description", "Country", "Quantity",
    "UnitPrice", "Hour", "Weekday", "Month",
    "Inventory", "CompetitorPrice", "TotalDemand"
}

ETL_DB2_SCHEMA = {
    "business_segment", "lead_type", "lead_behaviour_profile",
    "has_company", "has_gtin", "average_stock", "business_type",
    "avg_customer_age", "Quantity", "UnitPrice", "Hour",
    "Weekday", "Month", "Inventory", "CompetitorPrice", "TotalDemand"
}

def detect_target_db(columns):
    if columns == ETL_DB_SCHEMA:
        return "ETL_DB"
    elif columns == ETL_DB1_SCHEMA:
        return "ETL_DB_2"
    elif columns == ETL_DB2_SCHEMA:
        return "ETL_DB_3"
    return None

# =====================================================
# 3. SQL CONNECTION STRINGS
# =====================================================
SQL_CONNECTION_STRINGS = {
    "ETL_DB": (
        "Driver={ODBC Driver 17 for SQL Server};"
        "Server=aietl-sqldb-server-001.database.windows.net,1433;"
        "Database=ETL_DB;"
        "UID=CloudSA02c5ee85;"
        "PWD=Spoorthi@1234;"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=2000;"
    ),
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
    "ETL_DB_3": (
        "Driver={ODBC Driver 17 for SQL Server};"
        "Server=aietl-sqldb-server-001.database.windows.net,1433;"
        "Database=ETL_DB_3;"
        "UID=CloudSA02c5ee85;"
        "PWD=Spoorthi@1234;"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=2000;"
    )
}

TABLE_MAP = {
    "ETL_DB": "dbo.FACT_RETAIL",
    "ETL_DB_2": "dbo.FACT_DEMAND",
    "ETL_DB_3": "dbo.FACT_LEADS"
}

# =====================================================
# 4. CONNECT TO DATABASES
# =====================================================
connections = {}

print("\nConnecting to SQL databases...")
for db, cs in SQL_CONNECTION_STRINGS.items():
    conn = pyodbc.connect(cs, autocommit=False)
    connections[db] = conn
    cur = conn.cursor()
    cur.execute("SELECT DB_NAME()")
    print(f"Connected to {cur.fetchone()[0]} ✅")

# =====================================================
# 5. ETL CONFIG
# =====================================================
MAX_ROWS = 2000
CHUNK_SIZE = 500

print("\nStarting schema-based ETL...\n")

# =====================================================
# 6. ETL PROCESS
# =====================================================
for blob in container_client.list_blobs():
    print(f"Processing file: {blob.name}")

    # Skip unwanted files
    if blob.name != "Online_Retail_synthetic.csv":
        print(f"⏩ Skipping {blob.name}\n")
        continue

    blob_client = container_client.get_blob_client(blob.name)
    data = blob_client.download_blob().readall()

    df = (
        pd.read_excel(BytesIO(data))
        if blob.name.endswith(".xlsx")
        else pd.read_csv(BytesIO(data))
    )

    # =================================================
    # 🔑 CRITICAL FIX: NORMALIZE COLUMN NAMES
    # =================================================
    df.columns = (
        df.columns
        .astype(str)
        .str.replace("\u00a0", "", regex=False)  # remove NBSP
        .str.strip()                             # remove spaces
    )

    # Detect schema AFTER cleaning
    target_db = detect_target_db(set(df.columns))
    if not target_db:
        print("❌ Schema mismatch — skipped")
        print("Found :", set(df.columns))
        print("Expect:", ETL_DB1_SCHEMA | ETL_DB_SCHEMA | ETL_DB2_SCHEMA, "\n")
        continue

    # Limit rows
    df = df.head(MAX_ROWS)

    # Clean numeric data
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], None)
    df = df.where(pd.notnull(df), None)

    conn = connections[target_db]
    cursor = conn.cursor()
    cursor.fast_executemany = True

    table = TABLE_MAP[target_db]
    cols = ",".join(df.columns)
    placeholders = ",".join(["?"] * len(df.columns))

    sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"

    print(f"✅ Inserting into {table} ({target_db})")

    try:
        for start in range(0, len(df), CHUNK_SIZE):
            chunk = df.iloc[start:start + CHUNK_SIZE]
            cursor.executemany(sql, chunk.values.tolist())
            conn.commit()

        print(f"✔ Inserted {len(df)} rows\n")

    except Exception as e:
        conn.rollback()
        print("❌ Insert failed:", e, "\n")

# =====================================================
# 7. CLEANUP
# =====================================================
for conn in connections.values():
    conn.close()

print("🎯 ETL COMPLETED SUCCESSFULLY")
