#!/usr/bin/env bash
# Install + build the React frontend into webapp/frontend/dist
set -euo pipefail
cd "$(dirname "$0")/../webapp/frontend"
if [ ! -d node_modules ]; then
  npm install
fi
npm run build
echo "frontend built → $(pwd)/dist"
