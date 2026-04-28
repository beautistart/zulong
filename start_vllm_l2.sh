#!/bin/bash
source ~/vllm-env/bin/activate
export VLLM_USE_MODELSCOPE=true
vllm serve /mnt/d/AI/project/zulong_beta4/models/Qwen/Qwen3___5-0.8B-AWQ --port 8000 --host 0.0.0.0 --gpu-memory-utilization 0.5 --max-model-len 4096 --enable-auto-tool-choice --tool-call-parser qwen3_xml
