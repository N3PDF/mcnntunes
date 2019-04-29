# -*- coding: utf-8 -*-
"""
Performs MC tunes using Neural Networks
@authors: Stefano Carrazza & Simone Alioli
"""

import yaml, glob
from .tools import show, error

class ConfigError(ValueError): pass


class Config(object):
    """the yaml parser"""

    def __init__(self, content):
        """load lhe files"""
        self.content = content
        self.patterns = self.get('input', 'patterns')
        self.unpatterns = self.get('input', 'unpatterns')
        self.expfiles = self.get('input', 'expfiles')

        # Load and parse weight rules
        self.use_weights = False
        self.weightrules = []
        weightrules_list = self.get('input', 'weightrules')
        if weightrules_list == None:
            weightrules_list = []
        for rule in weightrules_list:
            try:
                pattern = rule['pattern']
                weight = rule['weight']
                
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
        if len(self.weightrules) != 0:
            self.use_weights = True

        self.seed = self.get('model', 'seed')
        self.scan = self.get('model', 'scan')
        if not self.scan:
            self.noscan_setup = self.get('model', 'noscan_setup')
        else:
            self.scan_setup = self.get('model', 'scan_setup')
        self.bounds = self.get('minimizer','bounds')
        self.restarts = self.get('minimizer','restarts')

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
            return cls(yaml.load(stream))
        except yaml.error.MarkedYAMLError as e:
            error('Failed to parse yaml file: %s' % e)