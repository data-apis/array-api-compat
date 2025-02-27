#!/usr/bin/env bash
# We cannot test cupy on CI so this script will test it manually. Assumes it
# is being run in an environment that has cupy and the array-api-tests
# dependencies installed
set -x
set -e

# Run the vendoring tests in this repo
pytest

tmpdir=$(mktemp -d)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH="$PYTHONPATH:$SCRIPT_DIR"

PYTEST_ARGS="--max-examples 200 -v -rxXfE --ci --hypothesis-disable-deadline"

cd $tmpdir
git clone https://github.com/data-apis/array-api-tests
cd array-api-tests

git submodule update --init

# store the hypothesis examples database in this directory, so that failures
# will be remembered across runs
mkdir -p $SCRIPT_DIR/.hypothesis
ln -s $SCRIPT_DIR/.hypothesis .hypothesis

export ARRAY_API_TESTS_MODULE=array_api_compat.cupy
export ARRAY_API_TESTS_VERSION=2024.12
pytest array_api_tests/ ${PYTEST_ARGS} --xfails-file $SCRIPT_DIR/cupy-xfails.txt "$@"
