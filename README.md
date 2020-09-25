# MC Neural Network Tunes

![Tests](https://github.com/scarrazza/mcnntune/workflows/Tests/badge.svg)

## Installation

```Shell
python setup.py develop
or
python setup.py install
```

## Programs

### mcnntemplate

Takes an input template card and variations card and generate variations.
```Shell
mcnntemplate myruncard_template.cmd myvariations.yml -o output
```

### mcnntune

Perform a MC tunes with Neural Networks.

```Shell
positional arguments:
  {preprocess,model,benchmark,tune,optimize}
                        sub-command help
    preprocess          preprocess input/output data
    model               fit NN model
    benchmark           check the goodness of the tuning procedure
    tune                provide final tune
    optimize            tune hyperparameters using the HyperOpt library
  runcard               the runcard file.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        the output folder
```
