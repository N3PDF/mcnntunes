# -*- coding: utf-8 -*-
"""
Generates custom runcards for tunes variations
"""
__author__ = "Stefano Carrazza & Simone Alioli"
__version__= "1.0.0"

import yaml, glob


class ConfigError(ValueError): pass


class Config(object):
    """the yaml parser"""

    def __init__(self, content):
        """load lhe files"""
        self.content = content

        try:
            self.yodafiles = []
            folder = content['input']['folder']
            for file in glob.glob('%s/*.yoda' % folder):
                self.yodafiles.append(file)
            if len(self.yodafiles) == 0:
                raise ConfigError('No yoda files found in %s' % folder)
            self.yodafiles.sort()
            print('Using %d files' % len(self.yodafiles))

            self.pattern = content['input']['pattern']
        except:
            raise ConfigError('Error input keyword not found in runcard.')

    @classmethod
    def from_yaml(cls, stream):
        """read yaml from stream"""
        try:
            return cls(yaml.load(stream))
        except yaml.error.MarkedYAMLError as e:
            raise ConfigError('Failed to parse yaml file: %s' % e)