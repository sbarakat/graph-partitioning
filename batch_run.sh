#!/bin/bash

ipython graph-partitioning-fennel-weights.py -- --num-partitions 4 --num-iterations 10 --prediction-model-cut-off 0.05 --one-shot-alpha 0.5 --restream-batches 10
ipython graph-partitioning-fennel-weights.py -- --num-partitions 4 --num-iterations 10 --prediction-model-cut-off 0.10 --one-shot-alpha 0.5 --restream-batches 10
ipython graph-partitioning-fennel-weights.py -- --num-partitions 4 --num-iterations 10 --prediction-model-cut-off 0.15 --one-shot-alpha 0.5 --restream-batches 10
ipython graph-partitioning-fennel-weights.py -- --num-partitions 4 --num-iterations 10 --prediction-model-cut-off 0.20 --one-shot-alpha 0.5 --restream-batches 10
ipython graph-partitioning-fennel-weights.py -- --num-partitions 4 --num-iterations 10 --prediction-model-cut-off 0.25 --one-shot-alpha 0.5 --restream-batches 10
ipython graph-partitioning-fennel-weights.py -- --num-partitions 4 --num-iterations 10 --prediction-model-cut-off 0.30 --one-shot-alpha 0.5 --restream-batches 10
ipython graph-partitioning-fennel-weights.py -- --num-partitions 4 --num-iterations 10 --prediction-model-cut-off 0.35 --one-shot-alpha 0.5 --restream-batches 10
