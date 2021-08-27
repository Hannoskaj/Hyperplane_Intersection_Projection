#!/usr/bin/env python
import setuptools
from setuptools.config import read_configuration


conf_dict = read_configuration("setup.cfg")
setuptools.setup(conf_dict)
