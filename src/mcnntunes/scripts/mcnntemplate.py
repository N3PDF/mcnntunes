# -*- coding: utf-8 -*-
"""
Generates custom runcards for tunes variations.
"""
import os
import argparse
import yaml
import itertools
import numpy as np

from jinja2 import Template


def main():

    # Parse the command-line arguments
    args = parse_args()

    # Open template
    with open(args.runcard_template, 'r') as f:
        template = f.read()

    # Open yaml file with parameter ranges/variations
    with open(args.variations, 'r') as f:
        variations = yaml.load(f, Loader=yaml.SafeLoader)

    # Make output directory
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Switch case
    if args.combinations:

        # Get all possible combination of parameter values
        a = []
        for key in variations:
            a += [variations[key]]
        configurations = list(itertools.product(*a))

    elif args.sampling:

        # Set random seed
        np.random.seed(args.seed)

        # Sample parameters
        values = np.zeros(shape=(len(variations), args.num_config))
        for i, key in enumerate(variations):
            assert len(variations[key]) == 2
            values[i,:] = np.random.uniform(variations[key][0], 
                                            variations[key][1],
                                            size=(1, args.num_config))

        configurations = [ [values[j,i] for j in range(len(variations))] for i in range(args.num_config)]

    # Save parameter configuration and build their runcards
    t = Template(template)
    for idx, c in enumerate(configurations):

        # Prepare the setup dictionary
        setup = dict()
        for i, item in enumerate(variations):
            setup[item] = c[i]

        # Create folder for the current run
        output_folder = os.path.join(args.output, f"{idx:04}")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Save runcard with parameters
        output_file = os.path.join(output_folder, "runcard.dat")
        print('Saving configuration %d in %s' % (idx,output_file))
        with open(output_file, 'w') as f:
            f.write(t.render(setup))

        # Save file with parameter values
        # using Professor format for compatibility
        output_file = os.path.join(output_folder, "params.dat")
        with open(output_file, 'w') as f:
            for i, item in enumerate(variations):
                f.write(f"{item} {c[i]}\n")

    print('Completed.')


def parse_args():
    """prepare the argument parser"""
    parser = argparse.ArgumentParser(
        description="Generates custom runcards for tunes variations.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='sub-command help')
    parser_comb = subparsers.add_parser('combinations', help='combination of parameter values provided in the variation runcard')
    parser_comb.set_defaults(combinations=True, sampling=False)

    parser_sampl = subparsers.add_parser('sampling', help='parameters sampled uniformly in a range')
    parser_sampl.add_argument('-n', '--num_config', help='number of parameter configurations', type=int, required=True)
    parser_sampl.add_argument('-s', '--seed', help='RNG seed', type=int, required=True)
    parser_sampl.set_defaults(combinations=False, sampling=True)

    parser.add_argument('runcard_template', help='the runcard template file.')
    parser.add_argument('variations', help='the variation runcard. (yaml)')
    parser.add_argument('-o','--output', help='the output folder for the generated files (default=output)', default='output')

    return parser.parse_args()


if __name__ == '__main__':
    main()
