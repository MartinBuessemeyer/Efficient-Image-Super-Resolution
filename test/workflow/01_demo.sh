#!/usr/bin/env bash

echo

test_def "run through demo"
(
    skip_on_hosts_other_than slurmsubmit

    export TARGET_TAG="my_project"
    export TARGET_SQSH="${TARGET_TAG}.sqsh"

    assert_cmd_successful srun -w ${DEBUG_HOST} --pty "${TEST_ROOT}/clean-test-setup.sh"

    assert_cmd_successful srun -w ${DEBUG_HOST} --pty ./scripts/build-image.sh
    assert_cmd_successful ./scripts/prepare.sh python main.py -a resnet18 --epochs 1

    sleep 3s
    if [ -z "${DEBUG_EMAIL}" ]; then
        NUM_GPU=1 assert_cmd_successful ./scripts/submit.sh runs/default/unnamed/last -w ${DEBUG_HOST}
    else
        NUM_GPU=1 EMAIL="${DEBUG_EMAIL}" assert_cmd_successful ./scripts/submit.sh runs/default/unnamed/last -w ${DEBUG_HOST}
    fi
    echo "Test training >submitted< at $(date +"%H:%M:%S (%Y-%m-%d)")."

    echo "Waiting for first log line."
    test_started=""
    while [ -z "${test_started}" ]; do
        sleep 3s
        if ! [ -z "$(grep "=> creating model 'resnet18'" runs/default/unnamed/last/logs/training.log)" ]; then
            test_started=true
        fi
    done
    echo "Test training >started< at $(date +"%H:%M:%S (%Y-%m-%d)")."

    echo "Waiting for training finish."
    test_finished=""
    while [ -z "${test_finished}" ]; do
        sleep 3s
        if ! [ -z "$(grep " * Acc" runs/default/unnamed/last/logs/training.log)" ]; then
            test_finished=true
        fi
    done
    echo "Test training >finished< at $(date +"%H:%M:%S (%Y-%m-%d)")."
)
test_end
