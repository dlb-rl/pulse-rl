#!/bin/bash

while getopts m: flag
do
    case "${flag}" in
        m) MODEL_VERSION=${OPTARG};;
    esac
done

# run the task
ray start --head
python -m luigi --module src.train_model.base TrainModel --model-version $MODEL_VERSION --local-scheduler

# get last command status so we can use it later
status=$?

ray stop --force

exit $status