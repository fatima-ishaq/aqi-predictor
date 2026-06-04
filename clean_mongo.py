"""
Clean MongoDB - Delete corrupted feature data
Run: python clean_mongo.py
"""

from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
CITY = os.getenv("CITY_NAME", "Karachi")

print(f"Connecting to MongoDB...")
client = MongoClient(MONGO_URI)
col = client["aqi_db"][f"features_{CITY.lower()}"]

# Count before deletion
count_before = col.count_documents({})
print(f"Documents before deletion: {count_before}")

# Delete all documents
result = col.delete_many({})
print(f"Deleted {result.deleted_count} documents")

# Verify
count_after = col.count_documents({})
print(f"Documents after deletion: {count_after}")

client.close()
print("Done!")