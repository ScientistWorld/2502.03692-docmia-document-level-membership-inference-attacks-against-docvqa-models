#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 0 GB
# Estimated disk usage after extraction: <1 MB
#
# This reproduction has no external downloads; the command below creates the
# small procedural dataset under data/.

set -e

cd /home/user

# The scaled reproduction uses a procedural DocVQA-style dataset so it is
# portable and does not require gated document downloads.
python data/synthetic_docvqa.py
