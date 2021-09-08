import os, sys
import pytest
import subprocess

import yoda

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

# Test mcnntemplate
def test_mcnntemplate():
    folder = "examples/templates"
    subprocess.check_call(str(f"mcnntemplate sampling -n 4 -s 2020 {folder}/pythia8_template.dat {folder}/pythia8_params.yml -o {folder}/sampling").split())
    subprocess.check_call(str(f"mcnntemplate combinations {folder}/pythia8_template.dat {folder}/pythia8_params.yml -o {folder}/combinations").split())

# Test mcnntunes-buildruns
def test_mcnntunes_buildruns():
    subprocess.check_call(str("mcnntunes-buildruns -n 5 -d ../mcnntunes_data/theory_input_3params_mpipt0_2.18/training_set_prof"
                        + " -f merged_3params.yoda --patterns ATLAS_2012_I1204784 ATLAS_2014_I1300647 --unpatterns RAW").split())

    # Check if output don't change with new releases
    # Open yodas
    for i in range(5):
        runA = yoda.read(f"output/run000{i}_merged_3params.yoda")
        runB = yoda.read(f"../mcnntunes_data/theory_input_3params_mpipt0_2.18/training_set/run000{i}_merged_3params.yoda")
        # Check for equal number of scatterplots
        assert len(runA) == len(runB)
        for run in runA:
            # Check for equal points
            assert runA[run].points() == runB[run].points()
            # Remove "Variations" annotation from old data, which is useless
            runB[run].rmAnnotation("Variations")
            # Remove "ErrorBreakdown" annotation from new data, for backwards compatibility
            runA[run].rmAnnotation("ErrorBreakdown")
            # Check for equal annotations
            assert runA[run].annotations() == runB[run].annotations() # same key
            for text in runA[run].annotations():
                if text != 'Run_Directory': # run directory may be different
                    assert runA[run].annotation(text) == runB[run].annotation(text) # same content
            del runB[run]
        # Check if runB contains a scatterplot that wasn't in runA
        assert len(runB) == 0

# List of example runcards for mcnntunes
def test_ptZ_Inverse():
    run_example("example_ptZ_Inverse.yml")


def test_ptZ_PerBin_CMAES():
    run_example("example_ptZ_PerBin_CMAES.yml")


def test_ptZ_PerBin_Grad():
    run_example("example_ptZ_PerBin_Grad.yml", options = ['preprocess', 'model', 'benchmark', 'tune'])
