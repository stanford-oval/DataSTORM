#!/bin/bash
# Start the llm-sandbox-datastorm container with proper volume mounts for STORM matplotlib visualization

# Stop and remove existing container if it exists
docker stop llm-sandbox-datastorm 2>/dev/null
docker rm llm-sandbox-datastorm 2>/dev/null

# Start the container with mounted volumes
# - Mount the SQL results directory where CSV files are stored
# - This allows the container to read CSV files for visualization
docker run -d \
  --name llm-sandbox-datastorm \
  -v "${DATASTORM_SQL_RESULTS_DIR:-$PWD/sql_results}":"${DATASTORM_SQL_RESULTS_DIR:-$PWD/sql_results}" \
  llm-sandbox

echo "llm-sandbox-datastorm container started with volume mounts:"
echo "  - ${DATASTORM_SQL_RESULTS_DIR:-$PWD/sql_results} (CSV files)"
