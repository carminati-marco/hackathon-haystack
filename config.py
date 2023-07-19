import os

# Configuration for Elastic
ELASTIC_SCHEME = os.getenv("ELASTIC_SCHEME")
ELASTIC_CLOUD_HOST = os.getenv("ELASTIC_CLOUD_HOST")
ELASTIC_PORT = os.getenv("ELASTIC_PORT", 9243)  # For cloud/hosted ES, 443
ELASTIC_API_ID = os.getenv("ELASTIC_API_ID")
ELASTIC_API_KEY = os.getenv("ELASTIC_API_KEY")
ELASTIC_INDEX = os.getenv("ELASTIC_INDEX", "test_gather_catalog")
