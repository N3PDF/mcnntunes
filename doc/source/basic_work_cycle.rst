Basic work cycle
================

The basic work cycle is subdivided in these steps:

* :ref:`sampling`
* :ref:`configure`
* :ref:`execute`

____________________

.. _sampling:

Generate a dataset of Monte Carlo runs
--------------------------------------

In the first step, you must create a dataset of Monte Carlo runs where each run has been generated with a different parameter configuration.

Once decided which parameters you need to tune, you are required to select a steering range for these parameters and insert them in a YAML file, for example:

.. code-block:: yaml

    # variations.yml
    # Parameter: [min, max]
    intrinsicKT: [1, 2.5]
    asMZisr: [0.12, 0.14]
    pT0isr: [0.5, 2.5]

Then, you need to create a runcard for your event generator, and set the value of the parameters above with the placeholder ``{{parameter_name}}``. This is an example using PYTHIA8:

.. code-block::

    ! runcard_template
    ! settings ...
    BeamRemnants:primordialKThard = {{intrinsicKT}} 
    SpaceShower:alphaSvalue = {{asMZisr}}
    SpaceShower:pT0Ref = {{pT0isr}}
    ! more settings ...

Now you can use

.. code-block::

    mcnntemplate sampling -n NUM_RUNS -s SEED runcard_template variations.yml

to sample ``NUM_RUNS`` parameter configurations uniformly within the steering ranges, and fill the runcard_template with these configurations.

Alternatively, the user can provide some possible values for each parameter and ``mcnntemplate`` can generate the parameter configurations by combining these values in all possible ways. The YAML file may be something like this:

.. code-block:: yaml

    # variations.yml
    # Parameter: [a, b, c, ...]
    intrinsicKT: [1, 1.5, 2, 2.5]
    asMZisr: [0.12, 0,13, 0.14]
    pT0isr: [0.5, 1.5, 2.5]

Then, running

.. code-block::

    mcnntemplate combinations runcard_template variations.yml

will compute the 36 possible combinations and fill the runcard_template with these configurations.

The output directory will be something like this:

.. code-block::

    output/0000/params.dat
    output/0000/runcard_template
    output/0001/params.dat
    output/0001/runcard_template
    ...
    output/NUM_RUNS/params.dat
    output/NUM_RUNS/runcard_template
    
Each runcard_template will use the parameters written in its params.dat file. Now run the generator (with the analyses that you prefer) for each of the templates and write the resulting result.yoda file in the corresponding folder.

In order to finalize the dataset, run

.. code-block::

    mcnntunes-buildruns -n NUM_RUNS -d output -f result.yoda -p params.dat --patterns INCLUDE_PATTERNS --unpatters EXCLUDE_PATTERNS -o training_set

The script will collect all histograms with a path that include ``INCLUDE_PATTERNS`` but not ``EXCLUDE_PATTERNS``. For example, ``INCLUDE_PATTERS`` may be ``ATLAS_2014_I1300647`` and ``EXCLUDE_PATTERNS`` may be ``RAW``. This step is required to embed the parameters in the yoda file and to convert histograms to scatterplots.

.. _configure:

Configure MCNNTUNES
-------------------

Now you need to configure MCNNTUNES. You can use the following template:

.. code-block:: yaml

    input:
        folders: folders containing the MC runs;
        patterns: list of patterns to look for in the histogram paths;
        unpatters: list of patterns to exclude;
        expfiles: list of files with the reference data;
        weightrules: a list of weight modifiers (optional)
            - pattern: it selects the histograms with that pattern in the path
                condition: see below
                weight: the weight (only 0 or 1 for the InverseModel)
            - ...

    # The condition subkey accept only:
    #    - one positive integer representing the index of the bin that we want to weight differently (the first bin is 1, not 0)
    #    - a list of two real number [a,b]. This will select all bins centered into the close interval [a,b].
    #      It's also possible to use '+inf' or '-inf' instead a real numbers.

    model:
        model_type: ('PerBinModel' or 'InverseModel')
        seed:
        noscan_setup:
            architecture: (optional, default [5, 5])
            actfunction: (optional, default 'tanh')
            optimizer: (optional, default "adam")
            optimizer_lr: (optional)
            initializer: (optional, default "glorot_uniform")
            epochs: (optional, default 5000)
            batch_size: (optional, default 16)
            data_augmentation: (optional, default False, only for 'InverseModel')
            param_estimator: (optional, only for 'InverseModel', 'SimpleInference', 'Median', 'Mean', default 'SimpleInference')

    # Minimizer is only for 'PerBinModel'
    minimizer:
        minimizer_type: ('CMAES' or 'GradientMinimizer' (experimental))
        bounds: boolean, bounds the results to be within the steering ranges (only for CMAES)
        restarts: number of minimization trials (only for CMAES)

Two different types of models are implemented. The `Per Bin` model parametrises the generator behaviour with fully-connected neural networks, and then fits the generator output to the experimental data using a minimizer. The `Inverse` model uses fully-connected neural networks, and tries to learn directly the parameter configuration that the generator needs to output a given result. For more information about the models, see https://arxiv.org/abs/20xx.xxxxxx.

Models are implemented with `Keras <https://keras.io/>`_, so you can use its activation functions, optimizers and initializers. The other keys under ``model`` are self-explanatory, except for ``param_estimator``: the `Inverse` model computes the tuning errors by generating a distribution of predictions within the experimental errors (see https://arxiv.org/abs/20xx.xxxxxx for more information). You can change the default parameter estimation from a simple inference to the mean or the median of this distribution.

Additional keys are required for more advanced usage, e.g. hyperparameter tuning (see :doc:`advanced usage <advanced_usage>`).

.. _execute:

Execute MCNNTUNES
-----------------

At first, perform the ``preprocess`` step and check if MCNNTUNES recognise the ``input`` keys successfully:

.. code-block::

    mcnntunes -o output preprocess runcard.yml

Then, perform the ``model`` step and check if MCNNTUNES recognise the ``model`` keys successfully:

.. code-block::

    mcnntunes -o output model runcard.yml

Finally, perform the ``tune`` step and check if MCNNTUNES recognise the ``minimizer`` keys successfully:

.. code-block::

    mcnntunes -o output tune runcard.yml

An HTML report with all the information about the tuning process will be created, and you can access it from ``output/index.html``.