# -*- coding: utf-8 -*-
"""
Performs MC tunes using Neural Networks
@authors: Stefano Carrazza & Simone Alioli
"""

import argparse, shutil, filecmp, logging
from runcardio import Config
from yodaio import Data
from nnmodel import NNModel
from minimizer import CMAES
from report import Report
from tools import make_dir, show, info, success, \
    error, __version__, __author__
import numpy as np


class App(object):

    def __init__(self):
        """reads the runcard and parse cmd arguments"""
        self.args = self.argparser().parse_args()
        make_dir(self.args.output)

        logging.basicConfig(format='%(message)s', filename='%s/output.log' % self.args.output,
                            filemode='w', level=logging.INFO)

        self.splash()

        with open(self.args.runcard, 'rb') as file:
            self.config = Config.from_yaml(file)
        if not self.args.load_from:
            shutil.copy(self.args.runcard, '%s/runcard.yml' % self.args.output)
        else:
            if not filecmp.cmp(self.args.runcard, '%s/runcard.yml' % self.args.output):
                error('Stored runcard has changed')

        np.random.seed(self.config.seed)

    def run(self):
        """main loop"""
        nn = NNModel()

        rep = Report(self.args.output)

        info('\n [======= MC data =======]')
        if self.args.load_from is not None:
            runs = Data.load('%s/runs.p' % self.args.load_from)
            nn.load('%s/model.h5' % self.args.load_from)
        else:
            # find MC run files
            self.config.discover_yodas()

            # open yoda files
            runs = Data(self.config.yodafiles, self.config.patterns, self.config.unpatterns)

            # saving data to file
            runs.save('%s/runs.p' % self.args.output)

            info('\n [======= Training NN model =======]')
            if not self.config.scan:
                nn.fit_noscan(runs.x_scaled, runs.y_scaled, self.config.noscan_setup)
            else:
                nn.fit_scan(runs.x_scaled, runs.y_scaled, self.config.scan_setup, self.args.parallel)

            # save model to disk
            nn.save('%s/model.h5' % self.args.output)

            rep.plot_model(nn, runs)

        info('\n [======= Experimental data =======]')
        expdata = Data(self.config.expfiles, ['/REF%s' % e for e in self.config.patterns],
                       self.config.unpatterns, expData=True)

        # check dims consistency
        if runs.y.shape[1] != expdata.y.shape[1]:
            error('Output dimension mismatch between MC and Experimental data')

        info('\n [======= Minimizing chi2 =======]')
        m = CMAES(nn, expdata, runs, self.config.bounds, self.args.output)
        result = m.minimize()

        best_x = result[0]*runs.x_std+runs.x_mean
        best_rel = np.abs(result[6]/result[0])
        best_std = best_x*best_rel

        rep.plot_minimize(m, best_x, result[0], runs)

        info('\n [======= Result Summary =======]')
        show('\n- Suggested best parameters for chi2/dof = %.6f' % result[1])

        display_output = { 'results' : [], 'version': __version__,
                           'chi2': result[1], 'dof': len(expdata.y[0]),
                           'loss': nn.loss[-1], 'scan': self.config.scan}
        for i, p in enumerate(runs.params):
            show('  =] (%e +/- %e) = %s' % (best_x[i], best_std[i], p))
            display_output['results'].append({'name':p,
                                              'x': str('%e') % best_x[i],
                                              'std': str('%e') % best_std[i]})

        info('\n [======= Generating report =======]')

        with open('%s/output.log' % self.args.output, 'rb') as f:
            display_output['raw_output'] = f.read()

        with open('%s/runcard.yml' % self.args.output, 'rb') as f:
            display_output['configuration'] = f.read()

        rep.save(display_output)

        success('\n [======= Completed =======]\n')

    def argparser(self):
        """prepare the argument parser"""
        parser = argparse.ArgumentParser(
            description="Perform a MC tunes with Neural Networks.",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        parser.add_argument('runcard', help='the runcard file.')
        parser.add_argument('--output', '-o', help='the output folder', default='output')
        parser.add_argument('--load-from', '-l', help='load model from folder')
        parser.add_argument('--parallel', dest='parallel', action='store_true',
                            help='use multicore during grid search', default=False)
        return parser

    def splash(self):

        show('  __  __  ____ _   _ _   _ _____                 ')
        show(' |  \\/  |/ ___| \\ | | \\ | |_   _|   _ _ __   ___ ')
        show(' | |\\/| | |   |  \\| |  \\| | | || | | | \'_ \\ / _ \\')
        show(' | |  | | |___| |\\  | |\\  | | || |_| | | | |  __/')
        show(' |_|  |_|\\____|_| \\_|_| \\_| |_| \\__,_|_| |_|\\___|')
        show('')

        show('  __version__: %s' % __version__)
        show('  __authors__: %s' % __author__)


def main():
    a = App()
    a.run()


if __name__ == '__main__':
    main()