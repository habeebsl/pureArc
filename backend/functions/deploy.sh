#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -x "$(command -v doctl)" ]]; then
  echo "Error: doctl is not installed or not in PATH"
  exit 1
fi

if [[ -f ".env" ]]; then
  set -a
  source .env
  set +a
fi

echo "Deploying functions project..."
doctl serverless deploy .

echo
echo "Function URL:"
doctl serverless functions get recommend/youtube_links --url
