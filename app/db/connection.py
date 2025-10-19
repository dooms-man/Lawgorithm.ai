# app/db/connection.py
import psycopg2
from app.config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS

conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASS
)
conn.autocommit = True
