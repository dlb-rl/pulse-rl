#!/bin/bash

while getopts m:v:d:i:w: flag
do
    case "${flag}" in
        m) MODEL_VERSION=${OPTARG};;
        v) DATASET_VERSION=${OPTARG};;
        d) DATASET_NAME=${OPTARG};;
        i) TRAIN_ID=${OPTARG};;
        w) WORKERS=${OPTARG};;
    esac
done

# run the task
ray start --head

python -m luigi --module src.train_model.evaluation Evaluation --num-workers ${WORKERS} \
--train-id ${TRAIN_ID} --local-eval  --datasets-name ${DATASET_NAME} \
--model-version ${MODEL_VERSION} --dataset-version ${DATASET_VERSION} \
--use-direct-estimator --use-custom-estimator  --local-scheduler

# get last command status so we can use it later
status=$?

ray stop --force

exit $status