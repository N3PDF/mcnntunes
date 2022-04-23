# -*- coding: utf-8 -*-
"""
Performs MC tunes using Neural Networks
"""
import numpy as np
import os, yoda
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from jinja2 import Environment, PackageLoader, select_autoescape
from mcnntunes.tools import make_dir, show
import mcnntunes.stats as stats


class Report(object):

    def __init__(self, path):
        """"""
        self.env = Environment(
            loader=PackageLoader('mcnntunes', 'templates'),
            autoescape=select_autoescape(['html'])
        )

        self.path = path
        make_dir('%s/plots' % self.path)

    def save(self, dictionary):
        """"""
        templates = ['index.html','data.html','model.html','benchmark.html',
                     'minimization.html','raw.html','config.html','optimize.html']

        for item in templates:
            template = self.env.get_template(item)
            output = template.render(dictionary)
            with open('%s/%s' % (self.path, item), 'w') as f:
                f.write(output)

        show('\n- Generated report @ file://%s/%s/index.html' % (os.getcwd(), self.path))

    def plot_CMAES_logger(self, logger):
        """"""
        logger.es.plot()
        plt.savefig(f'{self.path}/plots/minimizer.svg')
        plt.close()

    def plot_minimize(self, minimizer, best_x_unscaled, best_x_scaled, best_errors, runs, use_weights=False):
        """"""
        # plot 1d profiles
        N = 100 # points
        for dim in range(runs.x_scaled.shape[1]):
            d = np.linspace(np.min(runs.x_scaled[:,dim]), np.max(runs.x_scaled[:,dim]), N)
            res = np.zeros(N)
            if use_weights: # plot unweighted chi2 for more insight
                unw = np.zeros(N)
            xx = np.zeros(N)
            for i, p in enumerate(d):
                a = np.array(best_x_scaled)
                a[dim] = p
                res[i] = minimizer.chi2(a)
                if use_weights: # plot unweighted chi2 for more insight
                    unw[i] = minimizer.unweighted_chi2(a)
                xx[i] = runs.unscale_x(a)[dim]
            plt.figure()
            if not use_weights:
                plt.plot(xx, res, label='parameter variation', linewidth=2)
            else: # plot unweighted chi2 for more insight
                plt.plot(xx, res, label='parameter variation, weighted $\chi^2$/dof', linewidth=2)
                plt.plot(xx, unw, label='parameter variation, $\chi^2$/dof', linewidth=2)
            plt.axvline(best_x_unscaled[dim], color='r', linewidth=2, label='best value')
            plt.axvline(best_x_unscaled[dim]+best_errors[dim][1], linestyle='--', color='r', linewidth=2, label='1-$\sigma$')
            plt.axvline(best_x_unscaled[dim]-best_errors[dim][0], linestyle='--', color='r', linewidth=2)
            plt.legend(loc='best')
            plt.title('1D profiles for parameter %d - %s' % (dim, runs.params[dim]))
            plt.ylabel('$\chi^2$/dof')
            plt.xlabel('parameter')
            plt.grid()
            plt.yscale('log')
            plt.savefig('%s/plots/chi2_%d.svg' % (self.path, dim))
            plt.close()

    def plot_model(self, models, runs, data):
        """"""
        base = plt.cm.get_cmap('viridis')
        color_list = base(np.linspace(0, 1, len(models)))

        # scatter plot
        plt.figure()
        loss_runs = np.zeros(shape=(len(models), runs.x_scaled.shape[0]))
        diff_runs = np.zeros(shape=(len(models), runs.x_scaled.shape[0]))
        for i, model in enumerate(models):
            for r in range(runs.x_scaled.shape[0]):
                loss_runs[i,r] = model.model.evaluate(runs.x_scaled[r].reshape(1, runs.x_scaled.shape[1]),
                                 runs.y_scaled[r, i].reshape(1), verbose=0)
                diff_runs[i, r] = np.abs(model.predict(runs.x_scaled[r].reshape(1, runs.x_scaled.shape[1])).reshape(1)*runs.y_std[i]+runs.y_mean[i] - runs.y[r, i])/runs.y[r, i]
            #plt.plot([i]*runs.x_scaled.shape[0], loss_runs[i,:], 'o', color=color_list[i])

        bin_loss = []
        for model in models:
            bin_loss.append(model.loss[-1])

        plt.plot(bin_loss, color='k', label='avg. per bin')
        avg_loss = np.mean(bin_loss)
        std_loss = np.std(bin_loss)
        plt.axhline(avg_loss, color='r', linewidth=2, label='total')
        plt.axhline(avg_loss+std_loss, color='r', linestyle='--', label='std. dev.')
        plt.axhline(avg_loss-std_loss, color='r', linestyle='--')
        plt.legend(loc='best')
        plt.xlabel('bin')
        plt.ylabel('MSE')
        plt.title('Loss function for final model (bin-by-bin)')
        plt.grid()
        #plt.yscale('log')
        plt.savefig('%s/plots/model_loss.svg' % self.path)
        plt.close()

        # now plot vs parameters
        for param in range(runs.x.shape[1]):
            plt.figure()
            x = np.zeros(runs.x.shape[0]*len(models))
            y = np.zeros(runs.x.shape[0]*len(models))
            for bin in range(len(models)):
                #x = np.zeros(runs.x.shape[0])
                #y = np.zeros(runs.x.shape[0])
                for r in range(runs.x.shape[0]):
                    x[r+bin*runs.x.shape[0]] = runs.x[r][param]
                    y[r+bin*runs.x.shape[0]] = loss_runs[bin, r]
                #plt.scatter(x, y, marker='o', color=color_list[bin])
            plt.scatter(x, y, marker='o', color='k')
            plt.grid()
            plt.title('Loss function vs %s ' % runs.params[param])
            plt.xlabel('%s' % runs.params[param])
            plt.ylabel('MSE')
            plt.savefig('%s/plots/bounds_%d.svg' % (self.path, param))
            plt.close()

        # now plot the relative differences for the MC
        plt.figure()
        diffs = np.mean(runs.yerr/runs.y, axis=0)*100
        std = np.std(runs.yerr/runs.y, axis=0)*100
        plt.fill_between(range(len(diffs)), diffs-std, diffs+std, color='deepskyblue', edgecolor='deepskyblue', alpha=0.5)
        plt.plot(diffs, color='deepskyblue', linestyle='--', label='MC run mean error')
        nndiffs = np.mean(diff_runs, axis=1)*100
        nnstd = np.std(diff_runs, axis=1)*100
        plt.fill_between(range(len(nndiffs)), nndiffs-nnstd, nndiffs+nnstd, color='orange', edgecolor='orange', alpha=0.5)
        plt.plot(nndiffs, color='orange', linestyle='--', label='Model mean error')
        plt.title('Relative error for MC runs vs Model predictions')
        plt.ylabel('Relative error (%)')
        plt.xlabel('bin')
        plt.ylim([0, np.max(diffs+std)])
        plt.grid()
        plt.legend(loc='best')
        plt.savefig('%s/plots/errors.svg' % self.path)
        plt.close()

        return avg_loss

    def plot_data(self, data, predictions, runs, bestx, display):
        """"""
        hout = []
        ifirst = 0
        for i, hist in enumerate(data.plotinfo):
            size = len(hist['y'])
            rs = [item for item in runs.plotinfo if item['title'] == hist['title']]
            up = np.vstack([r['y'] for r in rs]).max(axis=0)
            dn = np.vstack([r['y'] for r in rs]).min(axis=0)
            reperr = np.mean([r['yerr'] for r in rs], axis=0)

            plt.figure()
            plt.subplot(211)

            plt.fill_between(hist['x'], dn, up, color='#ffff00')

            plt.errorbar(hist['x'], hist['y'], yerr=hist['yerr'],
                         marker='o', linestyle='none', label='data')

            plt.plot(hist['x'], predictions[ifirst:ifirst+size], '-', label='best model')

            plt.yscale('log')
            plt.legend(loc='best')
            plt.title(hist['title'])

            plt.subplot(212)

            plt.errorbar(hist['x'], hist['y']/hist['y'], yerr=hist['yerr']/hist['y'],
                         marker='o', linestyle='none', label='data')
            plt.plot(hist['x'], predictions[ifirst:ifirst+size]/hist['y'], '-', label='best model')

            plt.savefig('%s/plots/%d_data.svg' % (self.path, i))
            plt.close()

            # add to yoda
            h = yoda.Scatter2D(path=hist['title'])
            for t, p in enumerate(runs.params):
                h.setAnnotation(p, bestx[t])
            for p in range(size):
                h.addPoint(hist['x'][p], predictions[ifirst:ifirst+size][p], [hist['xerr-'][p], hist['xerr+'][p]])
            hout.append(h)

            # calculate chi2
            for j, element in enumerate(display):
                if element['name'] == hist['title']:
                    display[j]['model'] = stats.chi2(predictions[ifirst:ifirst+size], hist['y'],
                                                    np.square(hist['yerr'])+np.square(reperr))
                elif element['name'] == hist['title']+" (weighted)":
                    display[j]['model'] = stats.chi2(predictions[ifirst:ifirst+size], hist['y'],
                                                    np.square(hist['yerr'])+np.square(reperr), weights=hist['weight'])


            ifirst += size

        yoda_path = '%s/best_model.yoda' % self.path
        show('\n- Exporting YODA file with predictions in %s' % yoda_path)
        yoda.write(hout, yoda_path)

    def plot_correlations(self, corr):
        """"""
        plt.figure()
        plt.imshow(corr, cmap='Spectral', interpolation='nearest', vmin=-1, vmax=1)
        plt.colorbar()
        plt.savefig('%s/plots/correlations.svg' % self.path)
        plt.close()

    def plot_benchmark(self, benchmark_results):
        """"""
        # Iterate over the number of parameters
        for index1 in range(len(benchmark_results['single_closure_test_results'][0]['details'])):

            # Get parameter name
            param_name = benchmark_results['single_closure_test_results'][0]['details'][index1]['params']

            true_parameter = np.zeros(len(benchmark_results['single_closure_test_results']))
            relative_error = np.zeros(len(benchmark_results['single_closure_test_results']))

            # Iterate over  the number of closure tests
            for index2 in range(len(benchmark_results['single_closure_test_results'])):

                # Calculate the relative error
                true_parameter[index2] = benchmark_results['single_closure_test_results'][index2]['details'][index1]['true_params']
                relative_error[index2] = abs((benchmark_results['single_closure_test_results'][index2]['details'][index1]['predicted_params']-
                                         true_parameter[index2]) / true_parameter[index2]) * 100

            # Calculate mean value and std dev of the mean
            mean_error = np.mean(relative_error)
            mean_error_error = np.sqrt(np.var(relative_error)/relative_error.shape[0])

            # Plot
            plt.figure()
            plt.scatter(true_parameter, relative_error, color='k')
            plt.axhline(mean_error, color='r', linewidth=2, label='Mean')
            plt.axhline(mean_error+mean_error_error, color='r', linestyle='--', label='Std. dev. of the mean')
            plt.axhline(mean_error-mean_error_error, color='r', linestyle='--')
            plt.title("Relative difference on %s" % param_name)
            plt.xlabel("%s" % param_name)
            plt.ylabel('Relative difference (%)')
            plt.grid(True)
            plt.legend()
            plt.savefig('%s/plots/benchmark_%s.svg' % (self.path, param_name))
            plt.close()

    def plot_hyperscan_analysis(self, trials):
        """"""

        # List with available plots
        # for the HMTL report
        available_plots = []

        # Extract some data from the trials object
        data = {'iteration': [], 'loss': []}
        # Scan tuned settings using the first trial
        for conf in trials.trials[0]['configuration']:
            data[conf['key']] = []
            # Additional data if architecture was tuned
            if conf['key'] == 'architecture':
                architecture_info = {'nb_hidden_layers': [], 'average_units_per_layer': []}
        # Load data, while search for the best trial
        best_loss = 1000
        best_id = -1
        for trial in trials.trials:
            # Filter bad scans
            if trial['state'] != 2:
                continue
            # Load iteration number and loss
            data['iteration'].append(trial['tid'])
            data['loss'].append(trial['result']['loss'])
            # Search for the best trial
            if data['loss'][-1] < best_loss:
                best_loss = data['loss'][-1]
                best_id = data['iteration'][-1]
            # Load all tuned hyperparameters
            for conf in trial['configuration']:
                data[conf['key']].append(conf['value'])
            # Try to load additional data if architecture was tuned
            try:
                architecture_info['nb_hidden_layers'].append(len(data['architecture'][-1]))
                architecture_info['average_units_per_layer'].append(np.rint(np.mean(data['architecture'][-1])))
            except Exception:
                pass

        # Create a pandas dataframe for the trials,
        # and one for the best trial only
        df = pd.DataFrame(data=data)
        best_df = df[df['iteration'] == best_id]
        show('\n - Hyperparameter scan summary:\n')
        show(df)
        show('\n - Best trial:\n')
        show(best_df)

        # Plot loss against hyperparameters summary
        num_plots = len(data) - 2 # substract losses and iterations
        try: # subtract the architecture
            architecture_info
            num_plots -= 1
        except Exception:
            pass
        fig, ax = plt.subplots(1, num_plots, sharey=True, figsize=(3.2*num_plots, 5))
        current_ax = 0
        for key in data.keys():

            plt.figure()

            # Discrete hyperparameters
            if key in ('actfunction','initializer','optimizer','data_augmentation'):

                # Plot all trials
                sns.catplot(x=key, y='loss', kind='violin', cut=0.0, data=df, ax=ax[current_ax])
                sns.catplot(x=key, y='loss', kind='violin', cut=0.0, data=df)

                # Adjust plot settings
                if key == 'actfunction':
                    ax[current_ax].set_xlabel('activation function')
                    plt.xlabel('activation function')
                elif key == 'data_augmentation':
                    ax[current_ax].set_xlabel('data augmentation')
                    plt.xlabel('data augmentation')

            # Continuous hyperparameters
            elif key in ('batch_size','epochs','optimizer_lr'):

                # Plot all trials
                sns.relplot(x=key, y='loss', data=df, ax=ax[current_ax])
                sns.relplot(x=key, y='loss', hue='optimizer', data=df, style='optimizer')
                if key == 'batch_size':
                    ax[current_ax].set_xlabel('batch size')
                    plt.xlabel('batch size')
                elif key == 'optimizer_lr':
                    ax[current_ax].set_xlabel('learning rate')
                    ax[current_ax].set_xscale('log')
                    ax[current_ax].set_xlim(np.min(data['optimizer_lr']), np.max(data['optimizer_lr']))
                    plt.xlabel('learning rate')
                    plt.xscale('log')
                    plt.xlim(np.min(data['optimizer_lr']), np.max(data['optimizer_lr']))

                # Plot best trial
                sns.scatterplot(x=key, y='loss', color='red', data=best_df, ax=ax[current_ax], s=150)

            else:
                continue

            current_ax += 1
            plt.savefig(f'{self.path}/plots/hyper_scan_{key}.svg', bbox_inches='tight')
            available_plots.append(f'plots/hyper_scan_{key}.svg')

        # Save the figure
        fig.savefig(f'{self.path}/plots/hyper_scan.svg', bbox_inches="tight")
        plt.close()

        # Plot pairs
        plt.figure(figsize=(50, 50))
        try:
            slim_df = df.drop(['data_augmentation', 'iteration'], axis=1) # boolean type gives error
        except Exception:
            slim_df = df.drop('iteration', axis=1)
        sns.pairplot(slim_df)
        plt.savefig(f'{self.path}/plots/hyper_scan_pairplot.svg', bbox_inches='tight')
        available_plots.append('plots/hyper_scan_pairplot.svg')
        plt.close()

        # Plot architecture comparison
        try:
            plt.figure()
            archi_df = pd.DataFrame(data=architecture_info)
            df = df.join(archi_df)
            best_df = df[df['iteration'] == best_id]
            sns.relplot(x='average_units_per_layer', y='loss', row='nb_hidden_layers', data=df)
            sns.scatterplot(x='average_units_per_layer', y='loss', color='red', data=best_df, s=150)
            plt.xlabel('average units per layer')
            plt.savefig(f'{self.path}/plots/hyper_scan_architecture_best.svg', bbox_inches='tight')
            available_plots.append('plots/hyper_scan_architecture_best.svg')
            plt.close()
            plt.figure()
            sns.relplot(x='average_units_per_layer', y='loss', row='nb_hidden_layers', data=df, hue='optimizer', style='optimizer')
            plt.xlabel('average units per layer')
            plt.savefig(f'{self.path}/plots/hyper_scan_architecture.svg', bbox_inches='tight')
            available_plots.append('plots/hyper_scan_architecture.svg')
            plt.close()
        except Exception:
            pass

        return available_plots

    def plot_prediction_distribution(self, best_x, best_std, noisy_x, parameter_names):
        """"""

        # Create figure and axes
        fig, axes = plt.subplots(1, best_x.shape[0], figsize=(16,5))

        # Compute mean and median of the distribution
        mean = np.mean(noisy_x, axis=1)
        median = np.median(noisy_x, axis=1)

        # Plot histos and vertical lines
        for i in range(best_x.shape[0]):
            axes[i].hist(noisy_x[i,:], bins='auto', fill=True, density=True)
            axes[i].axvline(best_x[i], color='k', lw=2, label='Prediction')
            axes[i].axvline(best_x[i]+best_std[i], color='k', lw=1, ls='--', label='1 $\sigma$')
            axes[i].axvline(best_x[i]-best_std[i], color='k', lw=1, ls='--')
            axes[i].axvline(mean[i], color='b', ls='--', label='Mean')
            axes[i].axvline(median[i], color='r', ls='--', label='Median')
            axes[i].legend()
            axes[i].set_xlabel(parameter_names[i])
            axes[i].set_ylabel('$p(x)$')
        fig.suptitle('Distribution of predictions')

        # Save figure
        fig.savefig(f'{self.path}/plots/prediction_spread.svg', bbox_inches='tight')
        plt.close()
