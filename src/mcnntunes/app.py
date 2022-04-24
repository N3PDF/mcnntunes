# -*- coding: utf-8 -*-
"""
Performs MC tunes using Neural Networks
"""

# TODO: Improve the log management with the parallel search
# TODO: Improve HTML report when using InverseModel
# TODO: Add more options for the models

import numpy as np
import time, pickle, argparse, shutil, filecmp, logging, copy, yaml
from hyperopt import fmin as fminHyperOpt
from hyperopt import hp, tpe, Trials, STATUS_OK, space_eval
from hyperopt.mongoexp import MongoTrials
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from mcnntunes.runcardio import Config
from mcnntunes.yodaio import Data
from mcnntunes.nnmodel import get_model
from mcnntunes.minimizer import CMAES, GradientMinimizer
from mcnntunes.report import Report
from mcnntunes.tools import make_dir, show, info, success, error, log_check
import mcnntunes.stats as stats
import mcnntunes


class App(object):

    RUNS_DATA = '%s/data/runs.p'
    EXP_DATA = '%s/data/expdata.p'
    BENCHMARK_DATA = '%s/data/benchmark_data.p'

    def __init__(self):
        """reads the runcard and parse cmd arguments"""

        # Disable eager mode
        tf.compat.v1.disable_eager_execution()

        self.args = self.argparser().parse_args()
        make_dir(self.args.output)
        make_dir('%s/logs' % self.args.output)
        make_dir('%s/data' % self.args.output)

        if self.args.preprocess:
            outfile = '%s/logs/preprocess.log' % self.args.output
        elif self.args.model:
            outfile = '%s/logs/model.log' % self.args.output
        elif self.args.benchmark:
            outfile = '%s/logs/benchmark.log' % self.args.output
        elif self.args.tune:
            outfile = '%s/logs/tune.log' % self.args.output
        else:
            outfile = '%s/logs/optimize.log' % self.args.output

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
            # build and train model
            self.create_model(self.config.model_type, self.config.noscan_setup, output_path = self.args.output)
        elif self.args.benchmark:
            # check procedure goodness
            self.benchmark(self.config.model_type, self.config.minimizer_type, output_path = self.args.output)
        elif self.args.tune:
            # perform the tune and build the report
            self.tune()
        else:
            # do an hyperparameters scan with HyperOpt
            self.optimize()

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
            error('Number of output mismatch between MC runs and data.')

        # Prepare benchmark mode
        if self.config.use_benchmark_data:
            info('\n [======= Loading benchmark data =======]')

            # Loading benchmark data
            benchmark_data = Data(self.config.benchmark_yodafiles, self.config.patterns, self.config.unpatterns, self.config.weightrules, expData=False)

            # Save to disk
            benchmark_data.save(self.BENCHMARK_DATA % self.args.output)

        if self.config.model_type == 'PerBinModel':
            show('\n- You can now proceed with the {model} mode with bins=[1,%d]' % runs.y.shape[1])
        else:
            show('\n- You can now proceed with the {model} mode')

        # print chi2 to MC
        info('\n [======= Chi2 Data-MC =======]')

        summary = []
        # total chi2
        chi2 = []
        for rep in range(runs.y.shape[0]):
            chi2.append(stats.chi2(runs.y[rep], expdata.y, np.square(expdata.yerr)+np.square(runs.yerr[rep])))
        show('\n Total best chi2/dof: %.2f (@%d) avg=%.2f' % (np.min(chi2), np.argmin(chi2), np.mean(chi2)))
        summary.append({'name': 'TOTAL', 'min': np.min(chi2), 'mean': np.mean(chi2)})

        ifirst = 0
        for distribution in expdata.plotinfo:
            size = len(distribution['y'])
            chi2 = []
            for rep in range(runs.y.shape[0]):
                chi2.append(stats.chi2(runs.y[rep][ifirst:ifirst + size], distribution['y'],
                            np.square(distribution['yerr'])+np.square(runs.yerr[rep][ifirst:ifirst + size])))
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
                ifirst += size
                show(' |- %s (weighted): %.2f (@%d) avg=%.2f' % (distribution['title'], np.min(chi2), np.argmin(chi2), np.mean(chi2)))
                summary.append({'name': distribution['title']+" (weighted)", 'min': np.min(chi2), 'mean': np.mean(chi2)})

        pickle.dump(summary, open('%s/data/summary.p' % self.args.output, 'wb'))

        success('\n [======= Preprocess Completed =======]\n')

    def create_model(self, model_type, setup, output_path = None):
        """Build and train the NN models"""

        K.clear_session()

        info('\n [======= Model mode =======]')

        # Check if the model should be saved on disk
        if output_path is None:
            write_on_disk = False
        else:
            write_on_disk = True

        runs = Data.load(self.RUNS_DATA % self.args.output)

        info('\n [======= Training NN model =======]')

        nn = get_model(model_type, runs)
        nn.build_and_train_model(setup)
        if write_on_disk:
            nn.save_model_and_plots(f'{output_path}/{model_type}')

        success('\n [======= Training Completed =======]\n')

        return nn

    def benchmark(self, model_type, minimizer_type, output_path = None, nn = None, verbose = True):
        """
        This function performs a closure test of the tuning procedure: for each MC run in the benchmark dataset,
        it performs the Mcnntune tuning procedure using the MC run as the experimental data,
        and then compares the estimated parameters with the ones used during the run generation.
        """

        info('\n [======= Benchmark mode =======]')

        if not self.config.use_benchmark_data:
            error('Error: benchmark mode not enabled.\nTry loading some valid MC runs in the benchmark dataset.')

        # Check for a valid output path
        if output_path is None:
            write_on_disk = False
        else:
            write_on_disk = True

        # Load data and model
        runs = Data.load(self.RUNS_DATA % self.args.output)
        benchmark_data = Data.load(self.BENCHMARK_DATA % self.args.output)
        if nn is None:
            if write_on_disk:
                nn = get_model(model_type, runs)
                nn.search_and_load_model(f'{output_path}/{model_type}')
            else:
                error('Error: benchmark called without a model or a path')
        else:
            if model_type != nn.model_type:
                error('Error: model_type mismatch during benchmark call')

        # Initialize variables and folder
        benchmark_chi2 = np.zeros(benchmark_data.y.shape[0])
        benchmark_mean_relative_difference = np.zeros(benchmark_data.y.shape[0])
        benchmark_results = {}
        benchmark_results['single_closure_test_results'] = []
        if write_on_disk:
            make_dir(f'{output_path}/benchmark')

        # Calculate best parameters for each benchmark run
        for index in range(benchmark_data.y.shape[0]):

            if verbose:
                info(f'\n [======= Tuning benchmark run {index+1}/{benchmark_data.y.shape[0]}  =======]')

            if write_on_disk:
                current_output_path = f'{output_path}/benchmark/closure_test_{index+1}'
                make_dir(current_output_path)
            else:
                current_output_path = None

            # Get the predicted parameters, discriminating between direct and inverse models
            if model_type == 'PerBinModel':

                # Minimizer choice
                if minimizer_type == 'CMAES':
                    m = CMAES(runs, benchmark_data, nn,
                            current_output_path, truth_index=index,
                            useBounds=self.config.bounds, restarts=self.config.restarts)
                else:

                    # Clear previous GradientMinimizer instances while preserving the model
                    model_configs = [per_bin_nn.model.get_config() for per_bin_nn in nn.per_bin_nns]
                    model_weights = [per_bin_nn.model.get_weights() for per_bin_nn in nn.per_bin_nns]
                    K.clear_session()
                    for i in range(len(model_configs)):
                        nn.per_bin_nns[i].model = Sequential.from_config(model_configs[i])
                        nn.per_bin_nns[i].model.set_weights(model_weights[i])

                    m = GradientMinimizer(runs, benchmark_data, nn,
                            current_output_path, truth_index=index)

                # Get the predicted parameters
                best_x, best_std = m.minimize()
                best_std = np.mean(best_std, axis=1).squeeze() # mean between sx and dx error

            else:
                # For the InverseModel it's only an inference
                y = benchmark_data.y[index,:].reshape(1,benchmark_data.y.shape[1])
                y_err = benchmark_data.yerr[index,:].reshape(1,benchmark_data.yerr.shape[1])
                best_x, best_std = nn.predict(y, x_err = y_err, scaled_x = False, scaled_y = False)

            # Get the true parameters
            true_x = benchmark_data.x[index]

            # Compare the results with the true parameters
            benchmark_chi2[index] = stats.chi2(best_x,true_x,np.square(best_std))
            benchmark_mean_relative_difference[index] = np.mean(np.abs((best_x - true_x) / true_x)) * 100

            # Storing the results for later
            closure_test_results = []
            for parameter_index in range(runs.x.shape[1]):
                closure_test_results.append({'params': runs.params[parameter_index], 'true_params': true_x[parameter_index],
                                    'predicted_params': best_x[parameter_index], 'errors': best_std[parameter_index]})
            benchmark_results['single_closure_test_results'].append({'details': closure_test_results,
                                                                        'chi2': benchmark_chi2[index],
                                                                        'average_relative_difference': benchmark_mean_relative_difference[index]})

            # Printing the results
            if verbose:
                show("\n- {0:30} {1:30} {2:30} {3:30}".format("Params","True value","Predicted value","Error"))
                for row in closure_test_results:
                    show("  {0:30} {1:30} {2:30} {3:30}".format(row['params'],row['true_params'],
                                                                        row['predicted_params'],row['errors']))
                show("\n- Average chi2/dof: %f" % benchmark_chi2[index])
                show("\n- Average relative difference: %f %%" % benchmark_mean_relative_difference[index])

        # Storing benchmark results
        benchmark_results['chi2'] = np.mean(benchmark_chi2)
        benchmark_results['chi2_error'] = np.sqrt(np.var(benchmark_chi2)/benchmark_chi2.shape[0])
        benchmark_results['average_relative_difference'] = np.mean(benchmark_mean_relative_difference)
        benchmark_results['average_relative_difference_error'] = np.sqrt(np.var(benchmark_mean_relative_difference)
                                                                        / benchmark_mean_relative_difference.shape[0])
        if output_path == self.args.output:
            pickle.dump(benchmark_results, open('%s/data/benchmark.p' % self.args.output, 'wb'))

        # Printing benchmark results
        show("\n##################################################")
        show("\n- Total average chi2/dof: %f +- %f" % (benchmark_results['chi2'], benchmark_results['chi2_error']))
        show("\n- Total average relative difference: %f %% +- %f %%" %
                (benchmark_results['average_relative_difference'], benchmark_results['average_relative_difference_error']))

        success('\n [======= Benchmark Completed =======]\n')

        # Return the dictionary for the HyperOpt scan
        benchmark_results['status'] = STATUS_OK
        benchmark_results['loss'] = benchmark_results['average_relative_difference']
        benchmark_results['loss_variance'] = np.square(benchmark_results['average_relative_difference_error'])

        # Delete the details of each single closure test
        del benchmark_results['single_closure_test_results']

        return benchmark_results

    def tune(self):
        """Provide the final tune and build the HTML report"""

        info('\n [======= Tune mode =======]')

        # Load data and model
        runs = Data.load(self.RUNS_DATA % self.args.output)
        expdata = Data.load(self.EXP_DATA % self.args.output)
        nn = get_model(self.config.model_type, runs)
        nn.search_and_load_model(f'{self.args.output}/{self.config.model_type}')

        # Get the predicted parameters, discriminating between direct and inverse models
        if self.config.model_type == 'PerBinModel':

            # Minimizer choice
            if self.config.minimizer_type == 'CMAES':
                m = CMAES(runs, expdata, nn,
                        self.args.output, useBounds=self.config.bounds, restarts=self.config.restarts)
            else:
                m = GradientMinimizer(runs, expdata, nn, self.args.output)

            # Get the predicted parameters + stats
            best_x, best_std = m.minimize()
            chi2 = m.chi2(runs.scale_x(best_x))

        else:
            # For the InverseModel it's only an inference
            y = expdata.y[0,:].reshape(1,expdata.y.shape[1])
            y_err = expdata.yerr[0,:].reshape(1,expdata.yerr.shape[1])
            best_x, best_std, prediction_distribution = nn.predict(y, x_err = y_err, scaled_x = False,
                                            scaled_y = False, return_distribution = True,
                                            num_mc_steps = 100000)

        info('\n [======= Result Summary =======]')

        # print best parameters
        if self.config.model_type == 'PerBinModel':
            if self.config.use_weights:
                show('\n- Suggested best parameters for (weighted) chi2/dof = %.6f' % chi2)
            else:
                show('\n- Suggested best parameters for chi2/dof = %.6f' % chi2)
            for i, p in enumerate(runs.params):
                show('  =] %e [- %e, +%e] = %s' % (best_x[i], best_std[i][0], best_std[i][1], p))
        else:
            show('\n- Suggested best parameters:')
            for i, p in enumerate(runs.params):
                show('  =] (%e +/- %e) = %s' % (best_x[i], best_std[i], p))

        # print correlation matrix (if using PerBinModel + CMA-ES)
        if self.config.model_type == 'PerBinModel' and self.config.minimizer_type == 'CMAES':
            show('\n- Correlation matrix:')
            corr = m.get_fmin_output()[-2].sm.correlation_matrix
            for row in corr:
                show(row)

        info('\n [======= Building report =======]')

        # Start building the report
        rep = Report(self.args.output)
        display_output = {'results': [], 'version': mcnntunes.__version__, 'dof': runs.unweighted_dof,
                            'weighted_dof': runs.weighted_dof, 'model_type': self.config.model_type}

        # Add best parameters
        for i, p in enumerate(runs.params):
            param_details = {'name': p, 'x': str('%e') % best_x[i]}
            if display_output["model_type"] == "PerBinModel":
                param_details.update({'std':  str('%e') % best_std[i][0],
                                      'std2': str('%e') % best_std[i][1]})
            else:
                param_details.update({'std':  str('%e') % best_std[i]})
            display_output['results'].append(param_details)

        # Retrieve MC runs data
        display_output['summary'] = pickle.load(open('%s/data/summary.p' % self.args.output, 'rb'))

        # Switch case
        if display_output['model_type'] == 'PerBinModel':

            display_output['minimizer_type'] = self.config.minimizer_type

            # Add chi2
            if self.config.use_weights:
                display_output['weighted_chi2'] = chi2
                display_output['unweighted_chi2'] = m.unweighted_chi2(runs.scale_x(best_x))
            else:
                display_output['unweighted_chi2'] = chi2
            for i, element in enumerate(display_output['summary']):
                if element['name'] == 'TOTAL':
                    display_output['summary'][i]['model'] = display_output['unweighted_chi2']
                elif element['name'] == 'TOTAL (weighted)':
                    display_output['summary'][i]['model'] = display_output['weighted_chi2']

            # Add number of data-model plots
            display_output['data_hists'] = len(expdata.plotinfo)

            # Calculate prediction with best parameters (using nn model)
            up = nn.predict(best_x.reshape(1, best_x.shape[0]), scaled_x = False, scaled_y = False)

            # Make all plots needed in the report
            rep.plot_data(expdata, up, runs, best_x, display_output['summary'])
            rep.plot_minimize(m, best_x, runs.scale_x(best_x), best_std, runs, self.config.use_weights)
            if display_output['minimizer_type'] == 'CMAES':
                rep.plot_CMAES_logger(m.get_fmin_output()[-3])
                rep.plot_correlations(corr)
            display_output['avg_loss'] = rep.plot_model(nn.per_bin_nns, runs, expdata)

        else:
            # Plot distribution of prediction if using InverseModel
            rep.plot_prediction_distribution(best_x, best_std, prediction_distribution,
                                        [element['name'] for  element in display_output['results']])

        with open('%s/logs/tune.log' % self.args.output, 'r') as f:
            display_output['raw_output'] = f.read()
        with open('%s/runcard.yml' % self.args.output, 'r') as f:
            display_output['configuration'] = f.read()

        # Add benchmark results, if possible
        if self.config.use_benchmark_data:
            try:
                display_output['benchmark_results'] = pickle.load(open('%s/data/benchmark.p' % self.args.output, 'rb'))
                rep.plot_benchmark(display_output['benchmark_results'])
            except:
                show("\n- WARNING: Can't find benchmark results, please run benchmark mode first!")
                show("  Benchmark mode disabled.")

        # Add optimization results, if possible
        try:
            display_output['optimize_results'] = pickle.load(open(f'{self.args.output}/data/optimize.p', 'rb'))
            trials = pickle.load(open(f'{self.args.output}/data/trials.p', 'rb'))
            trials_is_defined = True
        except:
            show("\n- WARNING: Can't find optimize mode results, optimize mode disabled.")
            trials_is_defined = False
        if trials_is_defined:
            display_output['hyper_scan_plots'] = rep.plot_hyperscan_analysis(trials)

        rep.save(display_output)

        success('\n [======= Tune Completed =======]\n')

    def optimize(self):
        """Tune model hyperparameters with the HyperOpt library"""

        info('\n [======= Optimize mode =======]')

        if not self.config.enable_hyperparameter_scan:
            error('Error: no scan settings in the runcard')

        # Call the HyperOpt fmin function with the associated Trials object
        if self.config.enable_cluster:
            trials = MongoTrials(self.config.cluster_url, exp_key=self.config.cluster_exp_key)
        else:
            trials = Trials()
        best_config = fminHyperOpt(self.objective, space=self.config.model_scan_setup,
            algo=tpe.suggest, max_evals=self.config.max_evals, trials=trials)

        # Print the results
        best_config = space_eval(self.config.model_scan_setup, best_config)
        # Try to cast to int architecture, batch size and layer
        try:
            best_config['architecture'] = [int(size) for size in best_config['architecture']]
        except Exception:
            pass
        try:
            best_config['batch_size'] = int(best_config['batch_size'])
        except Exception:
            pass
        try:
            best_config['epochs'] = int(best_config['epochs'])
        except Exception:
            pass
        best_loss = trials.best_trial['result']['loss']
        best_loss_error = np.sqrt(trials.best_trial['result']['loss_variance'])
        show("\n- Best configuration:")
        for key, content in best_config.items():
            show(f"  ==] {key}: {content}")
        show(f"  Best loss: {best_loss}")
        show(f"  Best loss error: {best_loss_error}")

        # Add the configurations to the trials
        for i, trial in enumerate(trials.trials):
            configuration = {key: value[0] for key, value in trial['misc']['vals'].items() if len(value) > 0}
            configuration = space_eval(self.config.model_scan_setup, configuration)
            trials.trials[i]['configuration'] = [{'key': key, 'value': value} for key, value in configuration.items()]

        # Storing search results
        results = {'max_evals': self.config.max_evals,
                   'search_space': self.config.list_model_scan_setup,
                   'best_results': trials.best_trial['result'],
                   'trials': trials.trials}
        results['best_config'] = [{'key': key, 'value': value} for key, value in best_config.items()]
        pickle.dump(results, open(f'{self.args.output}/data/optimize.p', 'wb'))

        # Store the trials object
        pickle.dump(trials, open(f'{self.args.output}/data/trials.p', 'wb'))

        # Write a runcard with the best model
        original_runcard_dict = copy.deepcopy(self.config.content)
        del original_runcard_dict['hyperparameter_scan']
        original_runcard_dict['model']['noscan_setup'] = best_config
        with open(f'{self.args.output}/best_model.yml','w') as f:
            yaml.dump(original_runcard_dict, f, default_flow_style = False)

        success('\n [======= Optimize Completed =======]\n')

    def objective(self, configuration_dictionary):
        """The objective function for the hyperparameter scan"""

        # Check log
        log_check()

        # Fixing the configuration
        try:
            configuration_dictionary['epochs'] = int(configuration_dictionary['epochs'])
        except Exception:
            pass
        try:
            configuration_dictionary['batch_size'] = int(configuration_dictionary['batch_size'])
        except Exception:
            pass
        try:
            configuration_dictionary['architecture'] = [int(size) for size in configuration_dictionary['architecture']]
        except Exception:
            pass

        # Create the model and run a benchmark on it
        model = self.create_model(self.config.model_type, configuration_dictionary)
        benchmark_results = self.benchmark(self.config.model_type, self.config.minimizer_type, nn = model, verbose = False)

        return benchmark_results

    def argparser(self):
        """prepare the argument parser"""
        parser = argparse.ArgumentParser(
            description="Perform a MC tunes with Neural Networks.",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )

        subparsers = parser.add_subparsers(help='sub-command help')
        parser_preprocess = subparsers.add_parser('preprocess', help='preprocess input/output data')
        parser_preprocess.set_defaults(preprocess=True, model=False, benchmark=False, tune=False, optimize=False)

        parser_model = subparsers.add_parser('model', help='fit NN model')
        parser_model.set_defaults(model=True, preprocess=False, benchmark=False, tune=False, optimize=False)

        parser_benchmark = subparsers.add_parser('benchmark', help='check the goodness of the tuning procedure')
        parser_benchmark.set_defaults(preprocess=False, model=False, benchmark=True, tune=False, optimize=False)

        parser_tune = subparsers.add_parser('tune', help='provide final tune')
        parser_tune.set_defaults(model=False, preprocess=False, benchmark=False, tune=True, optimize=False)

        parser_optimize = subparsers.add_parser('optimize', help='tune hyperparameters using the HyperOpt library')
        parser_optimize.set_defaults(model=False, preprocess=False, benchmark=False, tune=False, optimize=True)

        parser.add_argument('runcard', help='the runcard file.')
        parser.add_argument('-o', '--output', help='the output folder', default='output')

        return parser

    def splash(self):

        show('  __  __  ____ _   _ _   _ _____                 ')
        show(' |  \\/  |/ ___| \\ | | \\ | |_   _|   _ _ __   ___ ')
        show(' | |\\/| | |   |  \\| |  \\| | | || | | | \'_ \\ / _ \\')
        show(' | |  | | |___| |\\  | |\\  | | || |_| | | | |  __/')
        show(' |_|  |_|\\____|_| \\_|_| \\_| |_| \\__,_|_| |_|\\___|')
        show('')

        show('  __version__: %s' % mcnntunes.__version__)


def main():
    a = App()
    a.run()


if __name__ == '__main__':
    main()
