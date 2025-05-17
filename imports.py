import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, friedmanchisquare

import opfunu
from enoppy.paper_based.ihaoavoa_2022 import TensionCompressionSpringProblem, CantileverBeamProblem
from enoppy.paper_based.moeosma_2023 import SpeedReducerProblem, CarSideImpactProblem
from enoppy.paper_based.pdo_2022 import PressureVesselProblem, ThreeBarTrussProblem, GearTrainProblem, IBeamProblem, TubularColumnProblem, PistonLeverProblem, CorrugatedBulkheadProblem, WeldedBeamProblem, ReinforcedConcreateBeamProblem
from mealpy import Optimizer, Multitask, FloatVar, PSO, WOA, GA, DE, AOA, SCSO

from helpers import *

#np.random.seed(12345)