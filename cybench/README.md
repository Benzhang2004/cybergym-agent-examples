# Cybench
commit: `6c3702c82d0e539aa5bbd85192e8ddaf96378fca`

## WARNING
Cybench agent requires docker privileged mode to enable docker-in-docker and special network configs, please be careful about directly running on your host server.
You can choose run in a dockerless environment with a separate user, or a standalone VM.

https://github.com/andyzorigin/cybench/blob/6272d1bfef5af39e7b4d2c687a8b8e3712dc2807/run_task.sh#L56

## Installation
```bash
docker build -t cybergym/cybench:latest .
```

## Run CyberGym
```bash
OPENAI_API_KEY=sk-...
CYBERGYM_DATA_DIR=...
OUT_DIR=...
MODEL=gpt-4.1-2025-04-14
SERVER_IP=...
SERVER_PORT=...
TASK_ID=...
python3 examples/agents/cybench/run.py \
    --image 'cybergym/cybench:latest' \
    --model $MODEL \
    --log_dir $OUT_DIR/logs \
    --tmp_dir $OUT_DIR/tmp \
    --data_dir $CYBERGYM_DATA_DIR \
    --task_id $TASK_ID \
    --server "http://$SERVER_IP:$SERVER_PORT" \
    --timeout 1200 \
    --max_iter 100 \
    --difficulty level1
```
