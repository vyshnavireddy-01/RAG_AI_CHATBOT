import requests

url = "http://127.0.0.1:5000/ingest"

data = {
    "url": "https://ealkay.com/sitemap.xml"
}

print("Starting ingestion...")

response = requests.post(url, json=data)

print("Server response:")
print(response.json())