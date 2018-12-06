import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from datasetup.utils import openFile
from datasetup.data_preprocessing import getPoints, filterPAR

file = 'imported/Cleaned_Dataset.csv'
data = openFile(file)
data = getPoints(data, 'ORIGIN')

data = data.set_index('TCID')

print(data.head())
data.to_csv('imported/Origin_NonFiltered.csv')

