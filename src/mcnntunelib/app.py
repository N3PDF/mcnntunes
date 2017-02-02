# -*- coding: utf-8 -*-
"""
Generates custom runcards for tunes variations
"""
__author__ = "Stefano Carrazza & Simone Alioli"
__version__= "1.0.0"

import argparse
from runcardio import Config
from yodaio import Data
from nnmodel import NNModel
from tools import make_dir


class App(object):

    def __init__(self):
        """reads the runcard and parse cmd arguments"""
        self.args = self.argparser().parse_args()
        with open(self.args.runcard,'rb') as file:
            self.config = Config.from_yaml(file)
        make_dir(self.args.output)

    def run(self):
        """main loop"""

        # open yoda files
        data = Data(self.config.yodafiles, self.config.pattern)

        # load data from yodas
        data.scan_space()

        # standardize
        data.standardize()

        # build fitting model
        nn = NNModel(data.x_scaled, data.y_scaled)

        if self.args.load_model is not None:
            nn.load(self.args.load_model)
        else:
            nn.fit()
            # save model to disk
            nn.save(self.args.output)

        import matplotlib.pyplot as plt
        plt.figure()
        for i in range(100):
            plt.plot(data.unscale_y(nn.predict(data.x_scaled)[i,:])/data.y[0,:], color='c')
        plt.plot(data.unscale_y(nn.predict(data.x_scaled)[0, :]) / data.y[0, :], 'o-')
        plt.show()

    def argparser(self):
        """prepare the argument parser"""
        parser = argparse.ArgumentParser(
            description="Perform a MC tunes with Neural Networks.",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        parser.add_argument('runcard', help='the runcard file.')
        parser.add_argument('--output', '-o', help='the output folder', default='output')
        parser.add_argument('--load-model', '-l', help='load model from file')
        return parser


def main():
    a = App()
    a.run()


if __name__ == '__main__':
    main()