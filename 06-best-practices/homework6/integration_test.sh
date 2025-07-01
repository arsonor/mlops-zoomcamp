#!/usr/bin/env bash

set -e  # Exit immediately on any error

# Step 1: Set environment variables
export INPUT_FILE_PATTERN="s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
export OUTPUT_FILE_PATTERN="s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"
export S3_ENDPOINT_URL="http://localhost:4566"

# Step 2: Start Localstack (optional, in case you want to handle this from script)
echo "🚀 Starting Localstack (if not already running)..."
docker-compose up -d

# Step 3: Wait a moment for Localstack to be ready
echo "⏳ Waiting for Localstack to be ready..."
sleep 3

# Step 4: Create the S3 bucket if not exists
echo "📦 Creating S3 bucket..."
aws --endpoint-url=$S3_ENDPOINT_URL s3 mb s3://nyc-duration || true

# Step 5: Run the integration test
echo "🧪 Running integration test..."
python integration_test.py

echo "✅ Done!"
