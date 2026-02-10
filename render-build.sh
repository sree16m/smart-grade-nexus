#!/usr/bin/env bash
# exit on error
set -o errexit

# Install Tesseract and its dependencies
apt-get update && apt-get install -y tesseract-ocr libtesseract-dev poppler-utils

# Install Python Dependencies
pip install -r requirements.txt
