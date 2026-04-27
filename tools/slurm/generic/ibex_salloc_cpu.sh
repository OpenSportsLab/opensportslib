#!/usr/bin/env bash

salloc \
  --job-name=osl_dbg \
  --partition=batch \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=6 \
  --mem=90G \
  --time=3:59:00
# Optional account flag:
# --account=conf-neurips-2026.05.15-ghanembs
