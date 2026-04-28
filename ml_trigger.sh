#!/bin/bash
LOGFILE="/home/aparna/ml_log.txt"
ALERT_JSON="/tmp/alert.json"

# If input is provided via pipe, use it. If not, use the argument.
if [ -t 0 ]; then
    cp "$1" "$ALERT_JSON"
else
    read INPUT
    echo "$INPUT" > "$ALERT_JSON"
fi

echo "------------------------------------------" >> $LOGFILE
echo "Trigger received at $(date)" >> $LOGFILE

/home/aparna/ml_venv/bin/python3 /home/aparna/ml_scripts/ml_model.py "$ALERT_JSON" >> "$LOGFILE" 2>&1
