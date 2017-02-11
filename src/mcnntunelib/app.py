# -*- coding: utf-8 -*-
"""
Performs MC tunes using Neural Networks
@authors: Stefano Carrazza & Simone Alioli
"""

import time
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

    RUNS_DATA = '%s/data/runs.p'

    def __init__(self):
        """reads the runcard and parse cmd arguments"""
        self.args = self.argparser().parse_args()
        make_dir(self.args.output)
        make_dir('%s/logs' % self.args.output)
        make_dir('%s/data' % self.args.output)

        if self.args.preprocess:
            outfile = '%s/logs/preprocess.log' % self.args.output
        elif self.args.model:
            outfile = '%s/logs/model_bin_%s.log' % (self.args.output,self.args.bin)
        else:
            outfile = '%s/logs/minimize.log' % self.args.output

        logging.basicConfig(format='%(message)s', filename=outfile,
                            filemode='w', level=logging.INFO)

        self.splash()

        with open(self.args.runcard, 'rb') as file:
            self.config = Config.from_yaml(file)
        if self.args.preprocess:
            shutil.copy(self.args.runcard, '%s/runcard.yml' % self.args.output)
        else:
            try:
                if not filecmp.cmp(self.args.runcard, '%s/runcard.yml' % self.args.output):
                    error('Stored runcard has changed')
            except OSError:
                error('Run preprocess first')

        np.random.seed(self.config.seed)

    def run(self):

        start_time = time.time()

        if self.args.preprocess:
            # first step
            self.preprocess()
        elif self.args.model:
            # perform fit per bin
            self.create_model()
        else:
            self.minimize()

        show(" --- %s seconds ---" % (time.time() - start_time))

    def preprocess(self):
        """Prepare and describe MC input data"""
        info('\n [======= Preprocess mode =======]')

        # search for yoda files
        self.config.discover_yodas()

        # open yoda files
        runs = Data(self.config.yodafiles, self.config.patterns, self.config.unpatterns)

        # saving data to file
        runs.save(self.RUNS_DATA % self.args.output)

        show('\n- You can now proceed with the {model} mode with bins=[1,%d]' % runs.y.shape[1])

        success('\n [======= Preprocess Completed =======]\n')

    def create_model(self):
        """main loop"""

        info('\n [======= Model mode =======]')

        runs = Data.load(self.RUNS_DATA % self.args.output)

        info('\n [======= Training NN model =======]')

        nn = NNModel()
        bin = self.args.bin
        if bin < 1 or bin > runs.y.shape[1]:
            error('Bin out of bounds [1,%d]' % runs.y.shape[1])

        make_dir('%s/model_bin_%d' % (self.args.output,bin))

        show('\n- Fitting bin %d' % bin)

        x = runs.x_scaled
        y = runs.y_scaled[:, bin-1]
        if not self.config.scan:
            nn.fit_noscan(x, y, self.config.noscan_setup)
        else:
            nn.fit_scan(x, y, self.config.scan_setup, self.args.parallel)

        # save model to disk
        nn.save('%s/model_bin_%d/model.h5' % (self.args.output, bin))
        nn.plot('%s/model_bin_%d' % (self.args.output, bin), x, y)

        success('\n [======= Minimize Completed =======]\n')

    def minimize(self):
        """"""
        info('\n [======= Minimize mode =======]')

        runs = Data.load(self.RUNS_DATA % self.args.output)

        info('\n [======= Experimental data =======]')
        expdata = Data(self.config.expfiles, ['/REF%s' % e for e in self.config.patterns],
                       self.config.unpatterns, expData=True)

        # check dims consistency
        if runs.y.shape[1] != expdata.y.shape[1]:
            error('Output dimension mismatch between MC and Experimental data')

        nns = []
        for bin in range(runs.y.shape[1]):
            nn = NNModel()
            nn.load('%s/model_bin_%d/model.h5' % (self.args.output, bin+1))
            nns.append(nn)

        info('\n [======= Minimizing chi2 =======]')
        m = CMAES(nns, expdata, runs, self.config.bounds, self.args.output)
        result = m.minimize()

        print result[0]
        best_x = result[0] * runs.x_std + runs.x_mean
        best_rel = np.abs(result[6] / result[0])
        best_std = best_x * best_rel

        info('\n [======= Result Summary =======]')
        show('\n- Suggested best parameters for chi2/dof = %.6f' % result[1])

        display_output = {'results': [], 'version': __version__,
                          'chi2': result[1], 'dof': len(expdata.y[0]),
                          'loss': 0, 'scan': self.config.scan}
        for i, p in enumerate(runs.params):
            show('  =] (%e +/- %e) = %s' % (best_x[i], best_std[i], p))
            display_output['results'].append({'name': p,
                                          'x': str('%e') % best_x[i],
                                          'std': str('%e') % best_std[i]})

        up = np.zeros(expdata.y.shape[1])
        for i, nn in enumerate(nns):
            up[i] = nn.predict(result[0].reshape(1, result[0].shape[0])).reshape(1)

        rep = Report(self.args.output)
        rep.plot_data(expdata, runs.unscale_y(up), runs)
        rep.plot_minimize(m, best_x, result[0], runs)

        display_output['data_hists'] = len(expdata.plotinfo)

        with open('%s/logs/minimize.log' % self.args.output, 'rb') as f:
            display_output['raw_output'] = f.read()

        with open('%s/runcard.yml' % self.args.output, 'rb') as f:
            display_output['configuration'] = f.read()

        rep.save(display_output)

        """
        rep = Report(self.args.output)

        if self.args.load_from is not None:
            #runs = Data.load('%s/runs.p' % self.args.load_from)
            #nn.load('%s/model.h5' % self.args.load_from)
            pass
        else:
            # find MC run files
            self.config.discover_yodas()

            # open yoda files
            runs = Data(self.config.yodafiles, self.config.patterns, self.config.unpatterns)

            # saving data to file
            runs.save('%s/runs.p' % self.args.output)

            info('\n [======= Training NN model =======]')
            if not self.config.scan:
                nn.fit_noscan(runs.x_scaled, runs.y_scaled[:,0], self.config.noscan_setup)
            else:
                nn.fit_scan(runs.x_scaled, runs.y_scaled[:,0], self.config.scan_setup, self.args.parallel)

            # save model to disk
            nn.save('%s/model.h5' % self.args.output)


        info('\n [======= Experimental data =======]')
        expdata = Data(self.config.expfiles, ['/REF%s' % e for e in self.config.patterns],
                       self.config.unpatterns, expData=True)

        rep.plot_model(nn, runs, expdata)



        info('\n [======= Minimizing chi2 =======]')
        m = CMAES(nn, expdata, runs, self.config.bounds, self.args.output)
        result = m.minimize()

        best_x = result[0]*runs.x_std+runs.x_mean
        best_rel = np.abs(result[6]/result[0])
        best_std = best_x*best_rel

        rep.plot_minimize(m, best_x, result[0], runs)

        info('\n [======= Result Summary =======]')
        show('\n- Suggested best parameters for chi2/dof = %.6f' % result[1])

        display_output = { 'results': [], 'version': __version__,
                           'chi2': result[1], 'dof': len(expdata.y[0]),
                           'loss': nn.loss[-1], 'scan': self.config.scan}
        for i, p in enumerate(runs.params):
            show('  =] (%e +/- %e) = %s' % (best_x[i], best_std[i], p))
            display_output['results'].append({'name':p,
                                              'x': str('%e') % best_x[i],
                                              'std': str('%e') % best_std[i]})

        info('\n [======= Generating report =======]')

        up = runs.unscale_y(nn.predict(result[0].reshape(1,result[0].shape[0])).reshape(expdata.y.shape[1]))
        rep.plot_data(expdata, up, runs)

        display_output['data_hists'] = len(expdata.plotinfo)

        with open('%s/output.log' % self.args.output, 'rb') as f:
            display_output['raw_output'] = f.read()

        with open('%s/runcard.yml' % self.args.output, 'rb') as f:
            display_output['configuration'] = f.read()

        rep.save(display_output)
        """
        success('\n [======= Minimize Completed =======]\n')

    def argparser(self):
        """prepare the argument parser"""
        parser = argparse.ArgumentParser(
            description="Perform a MC tunes with Neural Networks.",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )

        subparsers = parser.add_subparsers(help='sub-command help')
        parser_preprocess = subparsers.add_parser('preprocess', help='preprocess input/output data')
        parser_preprocess.set_defaults(preprocess=True, model=False, minimize=False)

        parser_model = subparsers.add_parser('model', help='fit NN model to bin')
        parser_model.add_argument('-b','--bin', help='the output bin to fit [1,NBINS]', type=int, required=True)
        parser_model.set_defaults(model=True, preprocess=False, minimize=False)

        parser_minimize = subparsers.add_parser('minimize', help='minimize and provide final tune')
        parser_minimize.set_defaults(model=False, preprocess=False, minimize=True)

        parser.add_argument('runcard', help='the runcard file.')
        parser.add_argument('-o', '--output', help='the output folder', default='output')
        parser.add_argument('-p','--parallel', dest='parallel', action='store_true',
                            help='use multicore during grid search cv', default=False)

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
