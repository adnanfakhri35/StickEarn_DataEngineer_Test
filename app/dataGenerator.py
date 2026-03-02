import random
import time
from datetime import datetime, timedelta
import uuid
import logging
import string

import polars as pl
import h3
from clickhouse_connect import get_client


TOTAL_ROWS = 1_000_000
BATCH_SIZE = 100_000
H3_LEVEL = 9  #10

LAT_MIN = -6.50
LAT_MAX = -5.90
LON_MIN = 106.50
LON_MAX = 107.10

CITY_CODES = [
    "JK-CENTRAL",
    "JK-SOUTH",
    "JK-WEST",
    "JK-EAST",
    "JK-NORTH"
]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def random_timestamp(start: datetime, end: datetime):
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=random_seconds)

def random_device_id():
    # 2 huruf + 2 angka + '-' + 2 huruf + 2 angka
    part1 = ''.join(random.choices(string.ascii_lowercase, k=2))
    part1 += ''.join(random.choices(string.digits, k=2))
    part2 = ''.join(random.choices(string.ascii_lowercase, k=2))
    part2 += ''.join(random.choices(string.digits, k=2))
    return f"{part1}-{part2}"

def generate_data(n_rows: int) -> pl.DataFrame:
    logging.info(f"Generating {n_rows} rows...")

    end_time = datetime.now()
    start_time = end_time - timedelta(days=365)

    df = pl.DataFrame({
        "time_stamp": [
            random_timestamp(start_time, end_time)
            for _ in range(n_rows)
        ],
        "deviceID": [random_device_id() for _ in range(n_rows)],
        "latitude": [random.uniform(LAT_MIN, LAT_MAX) for _ in range(n_rows)],
        "longitude": [random.uniform(LON_MIN, LON_MAX) for _ in range(n_rows)],
        "horizontal_accuracy": [round(random.uniform(10, 30), 2) for _ in range(n_rows)],
        "city_code": [random.choice(CITY_CODES) for _ in range(n_rows)]
    })

    return df

def add_h3_index(df: pl.DataFrame, level: int) -> pl.DataFrame:
    logging.info(f"Adding H3 index level {level}...")

    return df.with_columns(
        pl.struct(["latitude", "longitude"])
        .map_elements(
            lambda x: h3.latlng_to_cell(x["latitude"], x["longitude"], level),
            return_dtype=pl.Utf8
        )
        .alias("h3_index")
    )

def insert_batch_with_retry(client, table_name, data, column_names=None, max_retries=5, base_wait=180):
    """
    Insert batch ke ClickHouse dengan retry dan exponential backoff.

    Args:
        client: clickhouse_connect client
        table_name: nama table
        data: list of rows
        column_names: nama kolom
        max_retries: jumlah maksimal retry
        base_wait: waktu tunggu awal (detik), akan double tiap retry
    """
    attempt = 0

    while attempt < max_retries:
        try:
            client.insert(table_name, data, column_names=column_names)
            logging.info(f"Batch inserted successfully on attempt {attempt + 1}")
            return
        except Exception as e:
            wait_time = base_wait * (2 ** attempt)  # exponential backoff
            logging.error(f"Insert failed on attempt {attempt + 1}: {e}")
            logging.info(f"Retrying in {wait_time:.1f} seconds...")
            time.sleep(wait_time)
            attempt += 1

    raise Exception(f"Max retries exceeded ({max_retries}) for table {table_name}")


def main():

    client = get_client(
        host="clickhouse",
        port=8123,
        username="dev",
        password="dev123",
        database="default"
    )
    
    client.command("""
        CREATE TABLE IF NOT EXISTS default.mobility_events
        (
            time_stamp DateTime64(3),
            deviceID String,
            latitude Float64,
            longitude Float64,
            horizontal_accuracy Float32,
            city_code Enum8(
                'JK-CENTRAL' = 1,
                'JK-SOUTH'   = 2,
                'JK-EAST'    = 3,
                'JK-WEST'    = 4,
                'JK-NORTH' = 5
            ),
            h3_index LowCardinality(String)
        )
        ENGINE = ReplacingMergeTree(time_stamp)
        PARTITION BY toYYYYMM(time_stamp)
        ORDER BY (city_code, time_stamp, deviceID);
    """)

    df = generate_data(TOTAL_ROWS)
    df = add_h3_index(df, H3_LEVEL)

    logging.info("Starting batch insert...")

    for i in range(0, TOTAL_ROWS, BATCH_SIZE):
        batch = df.slice(i, BATCH_SIZE)
        insert_batch_with_retry(
            client,
            "mobility_events",
            batch.rows(),
            column_names=batch.columns
        )
        logging.info(f"Inserted batch {i} - {i + BATCH_SIZE}")

    logging.info("Batch insert has been processed.")


if __name__ == "__main__":
    main()