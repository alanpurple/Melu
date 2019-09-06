import pandas as pd
import numpy as np
import re
import json

movie_dat=pd.read_table('data/movies.dat','::',names=['title','genre'],index_col=0,engine='python')