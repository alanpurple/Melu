import pandas as pd
import numpy as np
import re
import json
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from urllib import parse
from data_models import Movie,User,Rating

movies_df=pd.read_table('data/movies.dat','::',names=['title_year','genre'],index_col=0,engine='python')
ratings_df=pd.read_table('data/ratings.dat','::',names=['UserID','MovieID','rating','timestamp'],engine='python')
users_df=pd.read_table('data/users.dat','::',names=['gender','age','occupation','zip'],index_col=0,engine='python')

movies_df['year']=movies_df.title_year.str[-5:-1].astype(int)
movies_df['title']=movies_df.title_year.str[:-7]
movies_df.drop('title_year',1)

server='10.102.40.94'
database='movielens'
driver='MySQL ODBC 8.0 Unicode Driver'
id='root'
pwd='wmind'

engine=create_engine('mysql+pyodbc://{}:{}@{}/{}?driver={}'.format(id,pwd,server,database,parse.quote_plus(driver)))

