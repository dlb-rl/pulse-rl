#!/bin/bash

# run the task
python -m luigi --module src.train_model.reward_pred_trainer TrainPredictor \
--model-version reward_pred --local-scheduler

# get last command status so we can use it later
status=$?

exit $status