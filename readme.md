# To run a finetune use the following command

```
python finetune_max.py --config /mnt/ceph_rbd/Process-Knowledge-Tracing/scripts/LoRa/max_interactions/llama3.2-20000.json
```

# To run inference on the finetuned model run:

```
python finetuned_inference.py --config /mnt/ceph_rbd/Process-Knowledge-Tracing/scripts/LoRa/max_interactions/llama3.2-20000.json --checkpoiont /mnt/ceph_rbd/Process-Knowledge-Tracing/scripts/LoRa/max_interactions/lamma3.2-3b/checkpoint-13000
```


# To run a DKT model 

```
python train_dkt.py --config path/to/configfile
```