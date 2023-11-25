import header as h
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import correlated_values
import uncertainties.unumpy as unp
import scipy.constants as constants

from matplotlib.legend_handler import (HandlerLineCollection,HandlerTuple)
from multiprocessing  import Process
