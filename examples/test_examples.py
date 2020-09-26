import os, sys
import pytest
import subprocess


def run_example(runcard, options = ['preprocess', 'model', 'benchmark', 'optimize', 'tune']):
    """Run an example."""

    # Write command line arguments in sys.argv
    sys.argv = ['', '', '', '', '']
    sys.argv[0] = 'mcnntunes'
    sys.argv[1] = '-o'
    sys.argv[2] = "examples/" + runcard[0:-4] + '_results'
    sys.argv[4] = "examples/" + runcard
    # Launch the script
    for sys.argv[3] in options:
        subprocess.check_call(sys.argv)


# List of example runcards
def test_ptZ_Inverse():
    run_example("example_ptZ_Inverse.yml")


def test_ptZ_PerBin_CMAES():
    run_example("example_ptZ_PerBin_CMAES.yml")


def test_ptZ_PerBin_Grad():
    run_example("example_ptZ_PerBin_Grad.yml", options = ['preprocess', 'model', 'benchmark', 'tune'])
