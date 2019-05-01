# -*- coding: utf-8 -*-
"""
Performs MC tunes using Neural Networks
@authors: Stefano Carrazza & Simone Alioli
"""

import time, pickle
import argparse, shutil, filecmp, logging
from .runcardio import Config
from .yodaio import Data
from .nnmodel import NNModel
from .minimizer import CMAES
from .report import Report
from .tools import make_dir, show, info, success, error, __version__, __author__
import mcnntunelib.stats as stats
import numpy as np


class App(object):

    RUNS_DATA = '%s/data/runs.p'
    EXP_DATA = '%s/data/expdata.p'

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

        # fix model seed
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

        # Print bins weighting
        self.config.print_weightrules()

        # search for yoda files
        self.config.discover_yodas()

        info('\n [======= Loading MC data =======]')

        # open yoda files
        runs = Data(self.config.yodafiles, self.config.patterns, self.config.unpatterns, self.config.weightrules)

        # saving data to file
        runs.save(self.RUNS_DATA % self.args.output)

        info('\n [======= Loading experimental data =======]')

        # loading experimental data
        expdata = Data(self.config.expfiles, self.config.patterns, self.config.unpatterns, self.config.weightrules, expData=True)

        # save to disk
        expdata.save(self.EXP_DATA % self.args.output)

        if runs.y.shape[1] != expdata.y.shape[1]:
            raise error('Number of output mismatch between MC runs and data.')

        show('\n- You can now proceed with the {model} mode with bins=[1,%d]' % runs.y.shape[1])

        # print chi2 to MC
        info('\n [======= Chi2 Data-MC =======]')

        summary = []
        # total chi2
        chi2 = []
        for rep in range(runs.y.shape[0]):
            chi2.append(stats.chi2(runs.y[rep], expdata.y, np.square(expdata.yerr)+np.square(runs.yerr[rep])))
            #chi2.append(np.mean(np.square((runs.y[rep]-expdata.y))/(np.square(expdata.yerr)+np.square(runs.yerr[rep]))))
        show('\n Total best chi2/dof: %.2f (@%d) avg=%.2f' % (np.min(chi2), np.argmin(chi2), np.mean(chi2)))
        summary.append({'name': 'TOTAL', 'min': np.min(chi2), 'mean': np.mean(chi2)})

        ifirst = 0
        for distribution in expdata.plotinfo:
            size = len(distribution['y'])
            chi2 = []
            for rep in range(runs.y.shape[0]):
                chi2.append(stats.chi2(runs.y[rep][ifirst:ifirst + size], distribution['y'],
                            np.square(distribution['yerr'])+np.square(runs.yerr[rep][ifirst:ifirst + size])))
                #chi2.append(np.mean(np.square((runs.y[rep][ifirst:ifirst + size] - distribution['y'])) / 
                                    #(np.square(distribution['yerr'])+np.square(runs.yerr[rep][ifirst:ifirst + size])) ))
            ifirst += size
            show(' |- %s: %.2f (@%d) avg=%.2f' % (distribution['title'], np.min(chi2), np.argmin(chi2), np.mean(chi2)))
            summary.append({'name': distribution['title'], 'min': np.min(chi2), 'mean': np.mean(chi2)})

        # print weighted chi2 to MC
        if self.config.use_weights:
            info('\n [======= Weighted Chi2 Data-MC =======]')

            # total chi2
            chi2 = []
            for rep in range(runs.y.shape[0]):
                chi2.append(stats.chi2(runs.y[rep], expdata.y, np.square(expdata.yerr)+np.square(runs.yerr[rep]), weights=runs.y_weight))
                #chi2.append(np.sum(np.square(runs.y_weight*(runs.y[rep]-expdata.y))/(np.square(expdata.yerr)+np.square(runs.yerr[rep])))
                                    #/runs.weighted_dof)
            show('\n Total best weighted chi2/dof: %.2f (@%d) avg=%.2f' % (np.min(chi2), np.argmin(chi2), np.mean(chi2)))
            summary.append({'name': 'TOTAL (weighted)', 'min': np.min(chi2), 'mean': np.mean(chi2)})

            ifirst = 0
            for distribution in expdata.plotinfo:
                size = len(distribution['y'])
                chi2 = []
                for rep in range(runs.y.shape[0]):
                    chi2.append(stats.chi2(runs.y[rep][ifirst:ifirst + size], distribution['y'],
                                np.square(distribution['yerr'])+np.square(runs.yerr[rep][ifirst:ifirst + size]),
                                weights=distribution['weight']))
                    #chi2.append(np.sum(np.square(distribution['weight']*(runs.y[rep][ifirst:ifirst + size] - distribution['y'])) /
                                    #(np.square(distribution['yerr'])+np.square(runs.yerr[rep][ifirst:ifirst + size])) ) / distribution['weighted_dof'])
                ifirst += size
                show(' |- %s (weighted): %.2f (@%d) avg=%.2f' % (distribution['title'], np.min(chi2), np.argmin(chi2), np.mean(chi2)))
                summary.append({'name': distribution['title']+" (weighted)", 'min': np.min(chi2), 'mean': np.mean(chi2)})

        pickle.dump(summary, open('%s/data/summary.p' % self.args.output, 'wb'))

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

        expdata = Data.load(self.EXP_DATA % self.args.output)

        nns = []
        for bin in range(runs.y.shape[1]):
            nn = NNModel()
            nn.load('%s/model_bin_%d/model.h5' % (self.args.output, bin+1))
            nns.append(nn)

        info('\n [======= Minimizing chi2 =======]')
        m = CMAES(nns, expdata, runs, self.config.bounds, self.args.output)
        result = m.minimize(self.config.restarts)

        logger = result[-3]
        best_x = result[0] * runs.x_std + runs.x_mean
        best_std = result[6] * runs.x_std

        info('\n [======= Result Summary =======]')
        
        # print best parameters
        if self.config.use_weights:
            show('\n- Suggested best parameters for (weighted) chi2/dof = %.6f' % result[1])
        else:
            show('\n- Suggested best parameters for chi2/dof = %.6f' % result[1])
        for i, p in enumerate(runs.params):
            show('  =] (%e +/- %e) = %s' % (best_x[i], best_std[i], p))

        # print correlation matrix
        show('\n- Correlation matrix:')
        corr = result[-2].correlation_matrix()
        for row in corr:
            show(row)

        # propose eigenvectors
        cov = np.zeros(shape=(len(corr),len(corr)))
        for i in range(cov.shape[0]):
            for j in range(cov.shape[1]):
                cov[i,j] = corr[i,j]*best_std[i]*best_std[j]
        eig, vec = np.linalg.eig(cov)
        replica = result[0] + (eig ** 0.5 * vec).T
        show('\n- Proposed 1-sigma eigenvector basis (Neig=%d):' % len(replica))
        for rep in replica:
            show(runs.unscale_x(rep))

        info('\n [======= Building report =======]')

        # Start building the report
        rep = Report(self.args.output)
        display_output = {'results': [], 'version': __version__, 'dof': len(expdata.y[0]), 'weighted_dof': runs.weighted_dof}

        # Add best parameters
        for i, p in enumerate(runs.params):
            display_output['results'].append({'name': p, 'x': str('%e') % best_x[i],
                                                'std': str('%e') % best_std[i]})

        # Add chi2
        if self.config.use_weights:
            display_output['weighted_chi2'] = result[1]
            display_output['unweighted_chi2'] = m.unweighted_chi2(result[0])
        else:
            display_output['unweighted_chi2'] = result[1]
        display_output['summary'] = pickle.load(open('%s/data/summary.p' % self.args.output, 'rb'))
        for i, element in enumerate(display_output['summary']):
            if element['name'] == 'TOTAL':
                display_output['summary'][i]['model'] = display_output['unweighted_chi2']
            elif element['name'] == 'TOTAL (weighted)':
                display_output['summary'][i]['model'] = display_output['weighted_chi2']

        # Calculate prediction with best parameters (using nn model)
        up = np.zeros(expdata.y.shape[1])
        for i, nn in enumerate(nns):
            up[i] = nn.predict(result[0].reshape(1, result[0].shape[0])).reshape(1)

        # Make all plots needed in the report
        rep.plot_correlations(corr)
        rep.plot_data(expdata, runs.unscale_y(up), runs, best_x, display_output['summary'])
        rep.plot_minimize(m, logger, best_x, result[0], best_std, runs, self.config.use_weights)
        display_output['avg_loss'] = rep.plot_model(nns, runs, expdata)

        # Add number of data-model plots
        display_output['data_hists'] = len(expdata.plotinfo)

        with open('%s/logs/minimize.log' % self.args.output, 'r') as f:
            display_output['raw_output'] = f.read()
        with open('%s/runcard.yml' % self.args.output, 'r') as f:
            display_output['configuration'] = f.read()

        rep.save(display_output)

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
