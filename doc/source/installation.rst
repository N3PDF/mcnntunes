Installation
====================

MCNNTUNES can be installed either with pip or from source:

* :ref:`installing-with-pip`
* :ref:`installing-from-source`
* :ref:`installing-YODA`

____________________

.. _installing-with-pip:

Installing with pip
-------------------

You can install MCNNTUNES directly from PyPI:

.. code-block:: bash

	pip install mcnntunes

The ``pip`` program will take care of all the required dependencies, except for YODA (view :ref:`installing-YODA`). Please note that MCNNTUNES requires Python 3.6 or greater.

.. _installing-from-source:

Installing from source
----------------------

If you prefer to install MCNNTUNES directly from source, clone the repository from GitHub:

.. code-block::

	git clone https://github.com/N3PDF/mcnntunes
	cd mcnntunes

Then, install the requirements:

.. code-block::

	pip install -r requirements.txt

The ``pip`` program will take care of all the required dependencies, except for YODA (view :ref:`installing-YODA`). Finally, install MCNNTUNES using ``pip``:

.. code-block::

	pip install .

.. _installing-YODA:

Installing external dependencies
--------------------------------

The `YODA library <https://yoda.hepforge.org/>`_ is not available from PyPI and must be installed manually. You can download it from the website and install it following standard GNU procedure, e.g.:

.. code-block::

	wget https://yoda.hepforge.org/downloads/?f=YODA-x.y.z.tar.gz -O YODA-x.y.z.tar.gz
	tar -xf YODA-x.y.z.tar.gz
	cd YODA-x.y.z
	./configure --prefix=$PREFIX
	make
	make install

where ``$PREFIX$`` is the installation path. If not specified, the default prefix is ``/usr/local/``. If you use a Conda package manager, you may be interested in using ``PREFIX=$CONDA_PREFIX``. Make sure to set up your environment properly so that the Python interpreter is able to import YODA.

In order to use the hyperparameter tuning procedure with a parallel search, you also need to install `MongoDB <https://www.mongodb.com/>`_.