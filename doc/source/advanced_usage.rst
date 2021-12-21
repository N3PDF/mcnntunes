Advanced usage
==============

This section presents some advanced functionalities.

* :ref:`benchmark`
* :ref:`hyperparameter-tuning`

___________________________________

.. _benchmark:

Validation with closure testing
-----------------------------------

If you take a set of run from the training set and move it in a separate folder, e.g. ``validation_set``, you can add it to the configuration runcard under the ``input`` key:

.. code-block:: yaml

    input:
        # ...
        benchmark_folders:
          - validation_set
        # ...

Now you have access to a performance assessment procedure based on `closure tests`:

.. code-block:: bash

    mcnntunes -o output benchmark runcard.yaml

For each run in the benchmark folders, the program will perform the tuning procedure with this run instead of the actual experimental data. Then, the obtained parameters are compared with the exact parameters used for generating the run, and a loss function is computed. For more information about the closure test procedure, see https://arxiv.org/abs/2010.02213.

.. _hyperparameter-tuning:

Hyperparameter tuning
---------------------

Deep learning algorithms present many hyperparameters that can heavily influence their performance. MCNNTUNES implements an hyperparameter tuning procedure using `HyperOpt <https://github.com/hyperopt/hyperopt>`_. In order to enable the hyperparameter optimization, you need to insert the corresponding settings in the configuration runcard:

.. code-block:: yaml

    hyperparameter_scan:
            max_evals: number of max  evaluation for the hyperparameter search
            model:
                architecture:
                actfunction:
                optimizer:
                epochs:
                batch_size:
                # other model settings...

You need to define the search space under the ``model`` key. It must be defined by means of specific function provided by HyperOpt (see the `official documentation <https://github.com/hyperopt/hyperopt/wiki/FMin>`_). You don't need to use each available key. An example of search space may be:

.. code-block:: yaml

    architecture: "hp.choice(’layers’, [[hp.quniform(f’size_{_}_2’,5,10,1) for _ in range(2)],
                                       [hp.quniform(f’size_{_}_3’,5,10,1) for _ in range(3)],
                                       [hp.quniform(f’size_{_}_4’,5,10,1) for _ in range(4)]])"
    optimizer: "hp.choice(’optimizer’, [’sgd’,’rmsprop’,’adagrad’,’adadelta’,’adam’,’adamax’])"
    optimizer_lr: "hp.loguniform(’learning_rate’, -10, -1)"

Now you can run the hyperparameter scan with

.. code-block:: bash

    mcnntunes -o output optimize runcard.yaml

At the end of the optimization procedure, a summary of all evaluations will be presented. Moreover, an in-depth analysis of the results is presented in the HTML report that ``mcnntunes tune`` builds. If you are satisfied with the best configuration, you can reconfigure MCNNTUNES with that configuration and re-execute the full basic work cycle.

.. note::

    Optionally, you can run a parallel search using the features of HyperOpt. This requires the instantiation of a MongoDB database that must be able to communicate with all the workers. Then, you must insert the database URL and an arbitrary `exp_key` to the configuration runcard:

    .. code-block:: yaml

        hyperparameter_scan:
                # settings...
                cluster:
                    url: URL
                    exp_key: EXP_KEY
                # settings...

    The program will send some work items to the database and wait for their results. These items corresponds to different trials for the hyperparameter scan. In order to evaluate these items, you need to launch some workers using e.g.

    .. code-block:: bash

        hyperopt-mongo-worker --exp-key=EXP_KEY --mongo=<host>[:port]/<db> --workdir=WORK_DIR

    See ``hyperopt-mongo-worker --help`` for more options. For more details about the HyperOpt implementation of the parallel search, see the official documentation.
