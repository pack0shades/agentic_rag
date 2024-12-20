#!/bin/bash

# Define the range and the number of processes to run at a time
START=2000
END=3000
NUM_PROCESSES=10
CHUNK_SIZE=$(( (END - START + 1) / NUM_PROCESSES ))

# Function to run a single process
run_process() {
  local FROM=$1
  local TO=$2
  python eval.py --qfrom "$FROM" --qto "$TO"
}

# Array to keep track of background processes
PIDS=()

# Loop to create and run processes in the background
for (( i=0; i<NUM_PROCESSES; i++ )); do
  FROM=$(( START + i * CHUNK_SIZE ))
  TO=$(( FROM + CHUNK_SIZE - 1 ))
  if [ "$TO" -gt "$END" ]; then
    TO=$END
  fi
  
  # Run the process in the background
  run_process "$FROM" "$TO" &
  
  # Save the PID of the process
  PIDS+=($!)
done

# Wait for all processes to complete
for PID in "${PIDS[@]}"; do
  wait "$PID"
done

echo "All processes have completed."
