@@ .. @@
 import os
 from dotenv import load_dotenv
 from langchain.text_splitter import RecursiveCharacterTextSplitter
 from openai import OpenAI
 import tiktoken

 # Load environment variables
 load_dotenv()

-# Set your OpenAI API key here or via environment variable
+# Get OpenAI API key from environment variable
 OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

-# Validate required environment variables
+# Validate required environment variable
 if not OPENAI_API_KEY:
     raise ValueError("Missing OPENAI_API_KEY environment variable. Please check your .env file.")

 # Files to process