#!/bin/bash

python trainer_pruning.py --batch 25 --epochs 125000 --prune_warmup 0 --dim_multipliers 1 1 --n_prune 0
python trainer_pruning.py --batch 25 --epochs 125000 --prune_warmup 0 --dim_multipliers 1 1 --n_prune 0
python trainer_pruning.py --batch 25 --epochs 125000 --prune_warmup 0 --dim_multipliers 1 1 --n_prune 0

# TODO: -----------
# python trainer_pruning.py --batch 25 --epochs 125000 --prune_warmup 20000 --dim_multipliers 2 2 --n_prune 22 --prune_type L2 --prune_perc 0.08 --prune_interv 1
# -----------------


# python trainer_pruning.py --batch 25 --epochs 125000 --prune_warmup 500 --dim_multipliers 2 2 --n_prune 22 --prune_type L2 --prune_perc 0.08 --prune_interv 1000
# python trainer_pruning.py --batch 25 --epochs 125000 --prune_warmup 500 --dim_multipliers 2 2 --n_prune 22 --prune_type L2 --prune_perc 0.08 --prune_interv 100
# python trainer_pruning.py --batch 25 --epochs 125000 --prune_warmup 500 --dim_multipliers 2 2 --n_prune 22 --prune_type L2 --prune_perc 0.08 --prune_interv 50
# python trainer_pruning.py --batch 25 --epochs 125000 --prune_warmup 500 --dim_multipliers 2 2 --n_prune 22 --prune_type L2 --prune_perc 0.08 --prune_interv 10
# python trainer_pruning.py --batch 25 --epochs 125000 --prune_warmup 500 --dim_multipliers 2 2 --n_prune 22 --prune_type L2 --prune_perc 0.08 --prune_interv 5
# python trainer_pruning.py --batch 25 --epochs 125000 --prune_warmup 500 --dim_multipliers 2 2 --n_prune 22 --prune_type L2 --prune_perc 0.08 --prune_interv 1



# python trainer_pruning.py --batch 25 --epochs 125000 --prune_warmup 1   --dim_multipliers 2 2 --n_prune 22 --prune_type L2 --prune_perc 0.08 --prune_interv 1
# python trainer_pruning.py --batch 25 --epochs 125000 --prune_warmup 5   --dim_multipliers 2 2 --n_prune 22 --prune_type L2 --prune_perc 0.08 --prune_interv 1
# python trainer_pruning.py --batch 25 --epochs 125000 --prune_warmup 10  --dim_multipliers 2 2 --n_prune 22 --prune_type L2 --prune_perc 0.08 --prune_interv 1
# python trainer_pruning.py --batch 25 --epochs 125000 --prune_warmup 50  --dim_multipliers 2 2 --n_prune 22 --prune_type L2 --prune_perc 0.08 --prune_interv 1
# python trainer_pruning.py --batch 25 --epochs 125000 --prune_warmup 100 --dim_multipliers 2 2 --n_prune 22 --prune_type L2 --prune_perc 0.08 --prune_interv 1
# python trainer_pruning.py --batch 25 --epochs 125000 --dim_multipliers 2 2 --n_prune 22 --arch aifnet --prune_type L2 --prune_perc 0.08 --prune_interv 1
# python trainer_pruning.py --batch 25 --epochs 125000 --dim_multipliers 2 2 2 2 --n_prune 13 --arch aifnet --prune_type L2 --prune_perc 0.12 --prune_interv 1

