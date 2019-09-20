import pandas as pd
from connect_db import Session,engine
from data_models import Movie

ratings_df=pd.read_table('data/ratings.dat','::',names=['user_id','movie_id','rate','ts'],engine='python')
ratings_df.drop('ts',1,inplace=True)
session=Session()
allids=[elem[0] for elem in session.query(Movie.id).all()]
ne_ratings_df=ratings_df[ratings_df.movie_id.isin(allids)]
ne_ratings_df.to_sql('rating',engine,if_exists='append',index=False,chunksize=50,method='multi')