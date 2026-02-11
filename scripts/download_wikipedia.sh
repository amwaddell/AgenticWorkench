#!/usr/bin/env bash
# Download Simple English Wikipedia pages+articles dump.
# Simple English Wikipedia is ~300 MB compressed — manageable for local dev.

set -euo pipefail

DUMP_URL="https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
DATA_DIR="data/raw/wikipedia"
OUTPUT_FILE="${DATA_DIR}/simplewiki-latest-pages-articles.xml.bz2"

echo "=== Simple English Wikipedia Download ==="

# Create directory
mkdir -p "${DATA_DIR}"

# Download (resume partial downloads with -C -)
if [ -f "${OUTPUT_FILE}" ]; then
    echo "File already exists: ${OUTPUT_FILE}"
    echo "Re-downloading with resume support..."
fi

echo "Downloading from: ${DUMP_URL}"
echo "Saving to:        ${OUTPUT_FILE}"
echo ""

curl -L -C - -o "${OUTPUT_FILE}" "${DUMP_URL}"

# Verify file exists and has non-trivial size (at least 10 MB)
if [ ! -f "${OUTPUT_FILE}" ]; then
    echo "ERROR: Download failed — file not found."
    exit 1
fi

FILE_SIZE=$(stat -f%z "${OUTPUT_FILE}" 2>/dev/null || stat --printf="%s" "${OUTPUT_FILE}" 2>/dev/null)
MIN_SIZE=$((10 * 1024 * 1024))  # 10 MB

if [ "${FILE_SIZE}" -lt "${MIN_SIZE}" ]; then
    echo "WARNING: File is only ${FILE_SIZE} bytes — may be incomplete."
    echo "Try running this script again to resume the download."
    exit 1
fi

echo ""
echo "Download complete!"
echo "File size: $(echo "scale=1; ${FILE_SIZE} / 1048576" | bc) MB"
echo "Location:  ${OUTPUT_FILE}"