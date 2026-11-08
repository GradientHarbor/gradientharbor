#!/usr/bin/env bash
# Copy ../../shared into the HarborCore bundle (the phone runs exactly the files the evals used).
set -euo pipefail
cd "$(dirname "$0")"
DST=Sources/HarborCore/Resources/shared
rm -rf "$DST" && mkdir -p "$DST"
cp -R ../../shared/fixtures ../../shared/prompts "$DST"/
cp ../../shared/tools.json ../../shared/redaction.json ../../shared/redaction_vectors.json ../../shared/router.json "$DST"/
echo "synced shared/ -> $DST"
