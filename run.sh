python -m torch.distributed.launch --nproc_per_node=24 \
           --master_port=12355 ./multil_thread_dividev2.py