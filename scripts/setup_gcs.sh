#!/bin/bash

# Setup Google Cloud Storage for Project SENTINEL
echo "=========================================="
echo "Setting up Google Cloud Storage for Project SENTINEL"
echo "=========================================="

# Configuration
export PROJECT_ID="bug-sync-467815"
export BUCKET_NAME="sentinel-kitti-data"
export REGION="us-central1"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI is not installed. Please install it first."
    echo "Visit: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Set project
echo "Setting GCP project to: $PROJECT_ID"
gcloud config set project $PROJECT_ID

# Check if authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "Please authenticate with Google Cloud:"
    gcloud auth login
fi

# Create bucket if it doesn't exist
if ! gsutil ls -b gs://$BUCKET_NAME &> /dev/null; then
    echo "Creating bucket: $BUCKET_NAME"
    gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$BUCKET_NAME
    
    # Set bucket permissions
    echo "Setting bucket permissions..."
    gsutil iam ch allUsers:objectViewer gs://$BUCKET_NAME
else
    echo "Bucket $BUCKET_NAME already exists"
fi

# Create folder structure in bucket
echo "Creating folder structure in bucket..."
gsutil -m mkdir -p gs://$BUCKET_NAME/sequences
gsutil -m mkdir -p gs://$BUCKET_NAME/models
gsutil -m mkdir -p gs://$BUCKET_NAME/results

# Set lifecycle rules for cost optimization (optional)
cat > /tmp/lifecycle.json << EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {
          "age": 90,
          "matchesPrefix": ["temp/"]
        }
      }
    ]
  }
}
EOF

gsutil lifecycle set /tmp/lifecycle.json gs://$BUCKET_NAME
rm /tmp/lifecycle.json

echo "=========================================="
echo "GCS setup complete!"
echo "Bucket: gs://$BUCKET_NAME"
echo "Project: $PROJECT_ID"
echo "=========================================="