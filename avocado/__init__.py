# __init__.py
# Contact: Jacob Schreiber <jmschr@cs.washington.edu>
#          William Noble   <wnoble@uw.edu>

"""
Avocado is deep tensor factorization model for learning a latent representation
of the human epigenome. This package contains tools to implement the model and
use it yourself.

This version of the package is deprecated. Version 0.4.1 is the last release
of this code before Avocado is re-implemented as a simpler package, and it
receives no further fixes.
See the README for the tested environment and the known issues.
"""

from .model import Avocado

__version__ = '0.4.1'
