import pandas as pd
import numpy as np
import re
import json
from data_models import Movie,User,Rating
from connect_db import engine,base,Session
import requests

movies_df=pd.read_table('data/movies.dat','::',names=['title_year','genre'],index_col=0,engine='python')
# ratings_df=pd.read_table('data/ratings.dat','::',names=['UserID','MovieID','rating','timestamp'],engine='python')
# users_df=pd.read_table('data/users.dat','::',names=['gender','age','occupation','zip'],index_col=0,engine='python')

movies_df['year']=movies_df.title_year.str[-5:-1].astype(int)
movies_df['title']=movies_df.title_year.str[:-7]
movies_df.drop('title_year',1,inplace=True)

omdb='http://www.omdbapi.com'

# daily 1000 limit, can use one of below
# params={'apikey':'f69c6afb'}
# params={'apikey':'a934a276'}
# params={'apikey':'7dad728b'}
# params={'apikey':'d58cf8f1'}
params={'apikey':'784e8ba1'}

directors=[]
actors=[]
rated=[]

session=Session()

# query=session.query(Movie)

# start_idx=0
stop_idx=0
failed_idx=[]

movies_df_c=movies_df[3770:].copy()

for row in movies_df_c.iterrows():
    params['t']=row[1].title
    params['year']=row[1].year
    returned=requests.get(omdb,params)
    if returned.status_code!=200:
        print('Error at row: {}'.format(row[0]))
        stop_idx=row[0]
        failed_idx.append(row[0])
        break
    result=returned.json()
    if 'Error' in result:
        print('failed at row: {}'.format(row[0]))
        failed_idx.append(row[0])
        continue
    directors.append(result['Director'].split(',')[0])
    actors.append(result['Actors'].split(',')[0])
    rated.append(result['Rated'])

if stop_idx>0:
    print('first error at: {}'.format(stop_idx))
    movies_df_c=movies_df[3770:stop_idx].copy()
else:
    print('not stopped')

movies_df_c.drop(failed_idx,inplace=True)

movies_df_c['director']=directors
movies_df_c['actor']=actors
movies_df_c['rated']=rated

movies_df_c.to_sql('movie',engine,if_exists='append',index_label='id',method='multi')

# base.metadata.create_all(engine)