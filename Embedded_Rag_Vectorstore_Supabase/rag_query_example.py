@@ .. @@
 import os
 from dotenv import load_dotenv
 from openai import OpenAI
 import psycopg2
 import numpy as np

 # Load environment variables
 load_dotenv()

-# Set your credentials
+# Get credentials from environment variables
 OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
 SUPABASE_DB_HOST = os.getenv("SUPABASE_DB_HOST")
 SUPABASE_DB_PORT = int(os.getenv("SUPABASE_DB_PORT", 5432))
 SUPABASE_DB_NAME = os.getenv("SUPABASE_DB_NAME")
 SUPABASE_DB_USER = os.getenv("SUPABASE_DB_USER")
 SUPABASE_DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")

 # Validate required environment variables
 required_vars = [OPENAI_API_KEY, SUPABASE_DB_HOST, SUPABASE_DB_NAME, SUPABASE_DB_USER, SUPABASE_DB_PASSWORD]
 if not all(required_vars):
     raise ValueError("Missing required environment variables. Please check your .env file.")