# -*- coding: utf-8 -*-
"""
Performs MC tunes using Neural Networks
"""
import yaml, glob
from hyperopt import hp
from mcnntunes.tools import show, error


class ConfigError(ValueError): pass


class Config(object):
    """
    This class parses the YAML runcard and loads the user settings.

    Usage:
        input:
            folders: folders containing the MC runs;
            patterns: list of patterns to look for in the MC runs histograms paths;
            unpatters: list of patterns to exclude;
            expfiles: list of files with the reference data;
            benchmark_folders: folders containing the MC runs used as benchmark for the tuning procedure;
            weightrules: a list of weight modifiers (optional)
                - pattern: it selects the histograms with that pattern in the path
                  condition: see below
                  weight: the weight (only 0 or 1 for the InverseModel)
                - ...

    The condition subkey accept only:
     - one positive integer representing the index of the bin that we want to weight differently (the first bin is 1, not 0)
     - a list of two real number [a,b]. This will select all bins centered into the close interval [a,b].
       It's also possible to use '+inf' or '-inf' instead a real numbers.

        model:
            model_type: 'PerBinModel' or 'InverseModel'
            seed:
            noscan_setup:
                architecture: (optional, default [5, 5])
                actfunction: (optional, default 'tanh')
                optimizer: (optional, default "adam")
                optimizer_rl: (optional)
                initializer: (optional, default "glorot_uniform")
                epochs: (optional, default 5000)
                batch_size: (optional, default 16)
                data_augmentation: (optional, default False, only for 'InverseModel')
                param_estimator:(optional, only for 'InverseModel', 'SimpleInference',
                                    'Median', 'Mean', default 'SimpleInference')

        minimizer: (only for 'PerBinModel')
            minimizer_type: 'CMAES' or 'GradientMinimizer' (experimental)
            bounds: boolean, bounds the results to be within the steering ranges (only for CMAES)
            restarts: number of minimization trials (only for CMAES)

        hyperparameter_scan:
            max_evals:
            cluster:
                url:
                exp_key:
            model:
                architecture:
                actfunction:
                optimizer:
                epochs:
                batch_size:

    """

    def __init__(self, content):
        """load the files"""
        self.content = content
        self.patterns = self.get('input', 'patterns')
        self.unpatterns = self.get('input', 'unpatterns')
        self.expfiles = self.get('input', 'expfiles')

        # Check for benchmark data
        try:
            self.get('input','benchmark_folders')
            self.use_benchmark_data = True
        except:
            self.use_benchmark_data = False

        # Check for weightrules
        self.weightrules = []
        try: # try to load weightrules
            weightrules_list = self.get('input', 'weightrules')
            self.use_weights = True
        except:
            self.use_weights = False

        # Parse weight rules, if present
        if self.use_weights:
            if weightrules_list == None: # just in case
                weightrules_list = []
            for rule in weightrules_list:
                try:
                    pattern = rule['pattern']
                    weight = rule['weight']
                    if weight < 0:
                        error('Error: negative weight found.')

                    # Analyze the condition
                    condition = rule['condition']
                    if isinstance(condition, int):
                        condition_type = 'bin_index'
                        if condition < 1:
                            error("Error: invalid bin index (use only integers > 0)")
                    elif len(condition)==2:
                        condition_type = 'interval'
                        for endpoint in condition:
                            if isinstance(endpoint, str):
                                if (endpoint != '+inf') and (endpoint != '-inf'):
                                    error("Error: unrecognised endpoint (use only numbers, '-inf' or '+inf')")
                    else:
                        error('Error: unrecognised condition format.')

                    # Build the right condition format
                    if condition_type == 'bin_index':
                        ruledict = {'pattern': pattern, 'condition_type': condition_type,
                        'bin_index': condition, 'weight': weight}
                    else:
                        ruledict = {'pattern': pattern, 'condition_type': condition_type,
                        'left_endpoint': condition[0], 'right_endpoint': condition[1], 'weight': weight}

                except:
                    error('Error: unrecognised weight rule format.')
                self.weightrules.append(ruledict)
            if len(self.weightrules) == 0: # check if the list was empty
                self.use_weights = False

        # Model subsection
        self.model_type = self.get('model', 'type')
        self.seed = self.get('model', 'seed')
        self.noscan_setup = self.get('model', 'noscan_setup')

        # Minimizer subsection
        if self.model_type == 'PerBinModel':
            self.minimizer_type = self.get('minimizer', 'type')
            if self.minimizer_type == 'CMAES':
                self.bounds = self.get('minimizer','bounds')
                self.restarts = self.get('minimizer','restarts')
        else:
            self.minimizer_type = None

        # Hyperparameters scan subsection
        try:
            self.max_evals = self.content['hyperparameter_scan']['max_evals']
            self.model_scan_setup = self.content['hyperparameter_scan']['model']
            self.list_model_scan_setup = [{'key': key, 'value': str(content)} for key, content in self.model_scan_setup.items()]
            self.enable_hyperparameter_scan = True
        except:
            self.enable_hyperparameter_scan = False

        # Parse scan settings, if present
        if self.enable_hyperparameter_scan:

            try:
                cluster_settings = self.content['hyperparameter_scan']['cluster']
                self.enable_cluster = True
            except:
                self.enable_cluster = False

            if self.enable_cluster:
                try:
                    self.cluster_url = cluster_settings['url']
                    self.cluster_exp_key = cluster_settings['exp_key']
                except:
                    error("Error: can't find proper cluster settings")

            for key, content in self.model_scan_setup.items():
                if 'hp.' in str(content):
                    self.model_scan_setup[key] = eval(content)

    def discover_yodas(self):
        try:
            self.yodafiles = []
            folders = self.content['input']['folders']
            for folder in folders:
                for f in glob.glob('%s/*.yoda' % folder):
                    self.yodafiles.append(f)
                if len(self.yodafiles) == 0:
                    error('No yoda files found in %s' % folder)
            self.yodafiles.sort()
            show('\n- Detected %d files with MC runs from:' % len(self.yodafiles))
            for folder in folders:
                show('  ==] %s' % folder)
        except:
            error('Error "input" keyword not found in runcard.')

        # Now do the same with the benchmark runs
        if self.use_benchmark_data:
            self.benchmark_yodafiles = []
            benchmark_folders = self.content['input']['benchmark_folders']
            for folder in benchmark_folders:
                for f in glob.glob('%s/*.yoda' % folder):
                    self.benchmark_yodafiles.append(f)
                if len(self.benchmark_yodafiles) == 0:
                    error('No yoda files found in %s' % folder)
            self.benchmark_yodafiles.sort()
            show('\n- Detected %d files with benchmark MC runs from:' % len(self.benchmark_yodafiles))
            for folder in benchmark_folders:
                show('  ==] %s' % folder)

    def get(self, node, key):
        """"""
        try:
            return self.content[node][key]
        except:
            error('Error key "%s" not found in node "%s"' % (key, node))

    def print_weightrules(self):
        """Print a nice summary of all weight rules"""
        show('\n- Checking for weight modifiers...')

        if len(self.weightrules) == 0:
            show('  No weight modifiers, all weights are set to 1.')
        else:
            show('  Detected %d modifiers:' % len(self.weightrules))
            for i, rule in enumerate(self.weightrules):
                    show('  ==] Rule %d:' % (i+1))
                    show('        Pattern: %s' % rule['pattern'])
                    if rule['condition_type'] == 'bin_index':
                        show('        Bin index: %d' % rule['bin_index'])
                    else:
                        left_endpoint = rule['left_endpoint']
                        right_endpoint = rule['right_endpoint']
                        show(f'        Left endpoint: {left_endpoint}')
                        show(f'        Right endpoint: {right_endpoint}')
                    show('        Weight: %.2f' % rule['weight'])

    @classmethod
    def from_yaml(cls, stream):
        """read yaml from stream"""
        try:
            return cls(yaml.load(stream, Loader=yaml.SafeLoader))
        except yaml.error.MarkedYAMLError as e:
            error('Failed to parse yaml file: %s' % e)
