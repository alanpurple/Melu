import pandas as pd
import numpy as np
import re
import json
from data_models import Movie,User,Rating
from connect_db import engine

movies_df=pd.read_table('data/movies.dat','::',names=['title_year','genre'],index_col=0,engine='python')
ratings_df=pd.read_table('data/ratings.dat','::',names=['UserID','MovieID','rating','timestamp'],engine='python')
users_df=pd.read_table('data/users.dat','::',names=['gender','age','occupation','zip'],index_col=0,engine='python')

movies_df['year']=movies_df.title_year.str[-5:-1].astype(int)
movies_df['title']=movies_df.title_year.str[:-7]
movies_df.drop('title_year',1)


