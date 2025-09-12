python examples/example.py --model-path /mnt/mcx/models/llama2-13b --model-name LLAMA2_13B --tp-degree 2 \
    --framework tensor > logfile/tensor.log

python examples/example.py --model-path /mnt/mcx/models/llama2-7b-chat --model-name llama2_7b --tp-degree 1 \
    --framework tensor > logfile/tensor.log

python examples/example.py --model-path /mnt/mcx/models/Yi-1.5-9B --model-name YI_9B --tp-degree 1 \
    --framework tensor > logfile/tensor.log

python examples/example.py --model-path /mnt/mcx/models/Yi-1.5-34B-Chat --model-name YI_34B --tp-degree 4 \
    --framework tensor > logfile/tensor.log

python examples/example.py --model-path /mnt/mcx/models/Qwen2.5-14B --model-name QWEN25_14B --tp-degree 2 \
    --framework tensor > logfile/tensor.log