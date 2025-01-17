#!/bin/bash
python make_hdf5.py --dataset E256 --batch_size 256 --data_root data --num_workers 8
python calculate_inception_moments.py --dataset E256_hdf5 --data_root data
