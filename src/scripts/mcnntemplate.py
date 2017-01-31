# -*- coding: utf-8 -*-
"""
Generates custom runcards for tunes variations
"""
__author__ = "Stefano Carrazza & Simone Alioli"
__version__= "1.0.0"

import os
import argparse
import yaml
import itertools
from jinja2 import Template


def main():
    args = parse_args()

    # open runcard
    with open(args.runcard_template, 'r') as f:
        template = f.read()

    # open yaml variation card
    with open(args.variations, 'r') as f:
        variations = yaml.load(f)

    a = []
    for key in variations: a += [variations[key]]
    configurations = list(itertools.product(*a))

    # mkdir
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # load and print templates
    t = Template(template)
    for idx, c in enumerate(configurations):
        setup = dict()
        for i, item in enumerate(variations):
            setup[item] = c[i]

        output_file = ('%s/%s_%d') % (args.output,args.runcard_template,idx)
        print('Saving configuration %d in %s' % (idx,output_file))
        with open(output_file, 'wb') as f:
            f.write(t.render(setup))

    print('Completed.')


def parse_args():
    """prepare the argument parser"""
    parser = argparse.ArgumentParser(
        description="Generates custom runcards for tunes variations.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('runcard_template', help='the runcard template file.')
    parser.add_argument('variations', help='the variation runcard. (yaml)')
    parser.add_argument('-o','--output', help='the output folder for the generated files (default=output)', default='output')
    return parser.parse_args()


if __name__ == '__main__':
    main()