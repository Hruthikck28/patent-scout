import psycopg2
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# --- CONFIGURATION ---
DB_URL = "postgresql://postgres.rdecqroqoecwimkrebhz:hruthikcharan@aws-1-ap-south-1.pooler.supabase.com:5432/postgres"

print("Loading Embedding Model...")
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# 1. Load and Clean the Real Kaggle Data
print("Reading Kaggle CSV...")
df = pd.read_csv("real_patents.csv")

# Drop rows missing crucial data
df = df.dropna(subset=['Abstract', 'Patent ID', 'Title', 'Priority Date'])

# IMPORTANT: Take a sample of 150 patents for a fast, free local demo. 
# You can increase this later once you know the pipeline works.
df = df.head(150) 

# Connect to Supabase
conn = psycopg2.connect(DB_URL)
cursor = conn.cursor()

print(f"Embedding and inserting {len(df)} real patents into PostgreSQL. Please wait...")
for index, row in df.iterrows():
    try:
        # Convert the actual patent abstract into a vector
        embedding = encoder.encode(str(row['Abstract'])).tolist()
        
        # Clean up the Assignee (some are missing in real data)
        assignee = str(row['Assignee']) if pd.notna(row['Assignee']) else "Unknown"
        
        cursor.execute("""
            INSERT INTO real_patents (patent_id, title, abstract, assignee, priority_date, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (patent_id) DO NOTHING;
        """, (row['Patent ID'], row['Title'], row['Abstract'], assignee, row['Priority Date'], embedding))
        
    except Exception as e:
        print(f"Skipping row {index} due to error: {e}")

conn.commit()
cursor.close()
conn.close()
print("✅ Real Data Ingestion Complete!")