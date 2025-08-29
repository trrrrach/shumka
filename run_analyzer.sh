#!/usr/bin/env bash

SOURCE="${1:-${ANALYZER_SOURCE}}"

if [ -z "$SOURCE" ]; then
  echo "Usage: $0 <path_to_source>"
  echo "       or set ANALYZER_SOURCE environment variable"
  exit 1
fi

python analyzer.py "$SOURCE" "${@:2}"
