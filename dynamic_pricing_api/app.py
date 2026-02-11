from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
app = Flask(__name__)
CORS(app)   # ✅ THIS LINE FIXES IT
import requests, threading
import pyodbc
import pandas as pd
import time
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ================== LOAD MODELS ==================
price_model = joblib.load("model/dynamic_pricing.joblib")
demand_model = joblib.load("model/demand_xgboost_model.pkl")

# 🔥 LOAD LABEL ENCODERS (REQUIRED)
country_encoder = joblib.load("model/le_country.pkl")
stock_encoder = joblib.load("model/le_stock.pkl")
df = pd.read_csv("model/customer_item_matrix.csv")
df.set_index("customer_id", inplace=True)
PRODUCT_NAMES = df.columns.tolist()
# ================== CONFIG ==================
PRICE_LEAKY_COLS = [
    "TotalDemand",
    "Revenue",
    "Profit",
    "Hour",
    "Weekday",
    "Month",
    "UnitPrice",
]

DEMAND_FEATURES = [
    'UnitPrice',
    'Discount',
    'PromotionFlag',
    'CompetitorPrice',
    'MarketTrend',
    'Inventory',
    'IsWeekend',
    'Weekday',
    'Month',
    'Quarter',
    'Country_enc',
    'StockCode_enc'
]
# ==========================
# 1. SQL CONNECTIONS
# ==========================
CONNECTION_STRINGS = {
    "ETL_DB": "Driver={ODBC Driver 17 for SQL Server};Server=tcp:aietl-sqldb-server-001.database.windows.net,1433;Database=ETL_DB;Uid=CloudSA02c5ee85;Pwd=Spoorthi@1234;Encrypt=YES;TrustServerCertificate=YES;Connection Timeout=300;",
    "ETL_DB_2": "Driver={ODBC Driver 17 for SQL Server};Server=tcp:aietl-sqldb-server-001.database.windows.net,1433;Database=ETL_DB_2;Uid=CloudSA02c5ee85;Pwd=Spoorthi@1234;Encrypt=YES;TrustServerCertificate=YES;Connection Timeout=300;",
    "ETL_DB_3": "Driver={ODBC Driver 17 for SQL Server};Server=tcp:aietl-sqldb-server-001.database.windows.net,1433;Database=ETL_DB_3;Uid=CloudSA02c5ee85;Pwd=Spoorthi@1234;Encrypt=YES;TrustServerCertificate=YES;Connection Timeout=300;"
}

connections = {db: pyodbc.connect(cs) for db, cs in CONNECTION_STRINGS.items()}
logging.info("Connected to all databases ✅")

# ==========================
# 2. SCHEMAS & TABLES
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
# 3. Row counters
# ==========================
total_rows_inserted = {"StockData": 0, "FACT_DEMAND": 0, "FACT_LEADS": 0}

# ==========================
# 4. INSERT FUNCTION (APPEND ONLY)
# ==========================
def insert_dataframe_append(df, table_name, conn, schema):
    global total_rows_inserted
    cursor = conn.cursor()
    df_to_insert = pd.DataFrame({col: df[col] if col in df.columns else None for col in schema})
    
    cols = ",".join(f"[{c}]" for c in schema)
    placeholders = ",".join("?" * len(schema))
    sql = f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})"
    
    rows = df_to_insert.astype(object).where(pd.notnull(df_to_insert), None).values.tolist()
    for row in rows:
        cursor.execute(sql, *row)
    conn.commit()
    total_rows_inserted[table_name] += len(rows)
    logging.info(f"Inserted {len(rows)} rows into {table_name}")

# ==========================
# 5. FETCH DATA FUNCTIONS
# ==========================
def fetch_products():
    url = "https://fakestoreapi.com/products"
    df = pd.DataFrame(requests.get(url).json())
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
# 6. ETL LOOP IN THREAD
# ==========================
SLEEP_INTERVAL = 10  # seconds

def etl_loop():
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

# Start ETL loop in a background thread


# ==========================
# 7. Flask endpoint
# ==========================
@app.route("/etl_stats")
def etl_stats():
    total_rows = sum(total_rows_inserted.values())
    return jsonify({
        "total_rows_inserted": total_rows,
        "breakdown": total_rows_inserted
    })

# ================== SIMILAR PRODUCTS (NAME-BASED) ==================
vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5)
)
PRODUCT_NAME_VECTORS = vectorizer.fit_transform(PRODUCT_NAMES)
# ================== PREPROCESSING ==================
def preprocess_price(data):
    df = pd.DataFrame(data)
    df = df.select_dtypes(include=["int64", "float64"])
    df = df.drop(columns=PRICE_LEAKY_COLS, errors="ignore")
    return df


def preprocess_demand(data):
    df = pd.DataFrame(data)

    # ---- ENCODE CATEGORICALS ----
    if "Country" not in df.columns or "StockCode" not in df.columns:
        raise ValueError("Country and StockCode are required")

    df["Country_enc"] = country_encoder.transform(df["Country"])
    df["StockCode_enc"] = stock_encoder.transform(df["StockCode"])

    # ---- DROP RAW COLUMNS ----
    df = df.drop(columns=["Country", "StockCode"])

    # ---- VALIDATE FEATURES ----
    missing = set(DEMAND_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {list(missing)}")

    return df[DEMAND_FEATURES]
# ================== UTILS ==================

def get_purchased_products(customer_id):
    row = df.loc[customer_id]
    return row[row == 1].index.tolist()


def get_similar_products(product_name, threshold=0.7):
    idx = PRODUCT_NAMES.index(product_name)
    target_vec = PRODUCT_NAME_VECTORS[idx]

    similarities = cosine_similarity(target_vec, PRODUCT_NAME_VECTORS)[0]

    return [
        PRODUCT_NAMES[i]
        for i, score in enumerate(similarities)
        if score > threshold and PRODUCT_NAMES[i] != product_name
    ]


def get_frequently_bought_together(product_name, threshold=0.5):
    target_vector = df[product_name].values.reshape(1, -1)

    similarities = cosine_similarity(target_vector, df.T.values)[0]

    return [
        PRODUCT_NAMES[i]
        for i, score in enumerate(similarities)
        if score > threshold and PRODUCT_NAMES[i] != product_name
    ]


# ================== ROUTES ==================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "Pricing & Demand Forecast API running"
    })


# -------- PRICE PREDICTION (UNCHANGED) --------
@app.route("/predict/price", methods=["POST"])
def predict_price():
    try:
        payload = request.get_json()
        X = preprocess_price(payload["inputs"])
        preds = price_model.predict(X)

        return jsonify({
            "predicted_unit_price": preds.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# -------- DEMAND PREDICTION (FIXED) --------
@app.route("/predict/demand", methods=["POST"])
def predict_demand():
    try:
        payload = request.get_json()
        X = preprocess_demand(payload["inputs"])
        preds = demand_model.predict(X)

        return jsonify({
            "predicted_quantity": preds.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/predict/customer-segmentation", methods=["POST"])
def customer_segmentation():
    try:
        payload = request.get_json()
        customer_id = payload.get("customer_id")

        if customer_id not in df.index:
            return jsonify({"error": "Customer ID not found"}), 404

        purchased_products = get_purchased_products(customer_id)

        similar_products = {}
        frequently_bought_together = {}

        for product in purchased_products:
            similar_products[product] = get_similar_products(product)
            frequently_bought_together[product] = get_frequently_bought_together(product)

        return jsonify({
            "customer_id": customer_id,
            "purchased_products": purchased_products,
            "similar_products": similar_products,
            "frequently_bought_together": frequently_bought_together
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ================== RUN ==================
if __name__ == "__main__":
    threading.Thread(target=etl_loop, daemon=True).start()
    app.run(
        port=8000,
        debug=True
    )
