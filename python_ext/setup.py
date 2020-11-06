#
from distutils.core import setup, Extension

YNN_module = Extension("YNN", sources = [ "YNN_core.c", "YNN.c" ])

setup(name        = "YNN",
      version     = "0.1",
      description = "YNN neural network library", 
      ext_modules = [ YNN_module ])
