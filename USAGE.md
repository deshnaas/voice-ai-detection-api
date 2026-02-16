# Usage Guide

## Step 1
Generate API Key.

## Step 2
Send POST request to /api/voice-detection

## Example Curl

curl -X POST https://your-api-url/api/voice-detection \
-H "x-api-key: YOUR_KEY" \
-H "Content-Type: application/json" \
-d '{
"language":"English",
"audioFormat":"mp3",
"audioBase64":"BASE64_STRING"
}'
