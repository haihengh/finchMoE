#!/bin/bash
cd /Users/john/Desktop/code/finchMoE/finchmoe-m1
for i in $(seq 1 25); do
  ./finchmoe-infer -R 9000 -m . --weights quant_clean/model_weights_quant.bin \
    --manifest quant_clean/model_weights_quant.json -e 0 --top-k 1 --no-think --rep-penalty 1.0 \
    >> /tmp/he_server.log 2>&1 &
  SRV=$!
  UP=0
  for t in $(seq 1 20); do
    sleep 3
    if curl -s -o /dev/null -m 2 http://127.0.0.1:9000/health; then UP=1; break; fi
    if ! kill -0 $SRV 2>/dev/null; then break; fi
  done
  if [ $UP -eq 1 ]; then
    echo "SERVER UP on attempt $i (pid $SRV)"
    wait $SRV; echo "SERVER EXITED rc=$?"
    exit 0
  fi
  kill $SRV 2>/dev/null
  echo "attempt $i: not up, waiting 45s..."
  sleep 45
done
echo "GAVE UP after 25 attempts"
