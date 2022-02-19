#!/usr/bin/env bash
python ./main.py --channel_name ising --p_ising 0.5 --lr 0.00075 --batches 10 --tag_name "checking_lr" &
python ./main.py --channel_name ising --p_ising 0.5 --lr 0.0015 --batches 10 --tag_name "checking_lr" &
python ./main.py --channel_name ising --p_ising 0.5 --lr 0.00045 --batches 10 --tag_name "checking_lr" &
python ./main.py --channel_name ising --p_ising 0.5 --lr 0.00025 --batches 10 --tag_name "checking_lr" &


