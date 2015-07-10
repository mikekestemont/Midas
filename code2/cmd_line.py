#!usr/bin/env python

import ConfigParser

def parse(config_path):
    """
    Parses the configuration file.
    Input: path to config-file
    Returns: a parameter dict
    """

    config = ConfigParser.ConfigParser()
    config.read(config_path)

    param_dict = dict()

    for section in config.sections():
        for name, value in config.items(section):

            if value.isdigit():
                value = int(value)
            elif value == "True":
                value = True
            elif value == "False":
                value = False

            param_dict[name] = value

    return param_dict