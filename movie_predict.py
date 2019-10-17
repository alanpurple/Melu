import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import metrics,Model,layers,Sequential,losses,optimizers,utils
from connect_db import Session,engine
from data_models import Movie,User,Rating
import json
from melu_model import MeluGlobal,MeluLocal
from sqlalchemy import func
from math import floor

MOVIE_MIN_YEAR=1919
MOVIE_MAX_YEAR=2000
MAX_USER_ID=6040

def main():
    session=Session()
    # query with condition? alternative
    all_users=session.query(User).all()
    all_movies=session.query(Movie).all()

    user_rating_counts=session.query(Rating.user_id,func.count(Rating.user_id)).group_by(Rating.user_id).all()

    # user with more than 40 ratings
    user_filtered=filter(lambda x: x[1]>45,user_rating_counts)
    actual_users_index=[elem[0] for elem in user_filtered]

    actor_dict,director_dict,rated_dict,genre_dict=get_movie_dict('movie_dict.json')
    #author_dict,publisher_dict=get_book_dict('book_dict.json')
    
    with open('movie_user_zipcodes.json','r') as f:
        zipcodes=json.load(f)
    zipcode_dict=dict(zip(zipcodes,range(len(zipcodes))))

    all_users_id=[elem.id for elem in all_users]
    all_users_data=[{'gender':elem.gender,'occupation':elem.occupation,'age':elem.age,'zipcode':elem.zipcode} for elem in all_users]

    all_users_df=pd.DataFrame(all_users_data,index=all_users_id)

    # occupation doesn't need hashing
    occu_dict_size=all_users_df.occupation.max()+1

    all_users_df.gender=(all_users_df.gender=='M').astype(int)
    all_users_df.zipcode=all_users_df.zipcode.apply(lambda x: zipcode_dict[x])

    user_ages=sorted(all_users_df.age.unique())
    # age may be quantifiable, but every person in their age periods has their own culture and style 
    age_dict=dict(zip(user_ages,range(len(user_ages))))

    all_users_df.age=all_users_df.age.apply(lambda x:age_dict[x])

    all_movies_id=[elem.id for elem in all_movies]
    all_movies_data=[{
        'year':elem.year,'actor':elem.actor,'title':elem.title,'rated':elem.rated,
        'director':elem.director,'genre':elem.genre
        } for elem in all_movies]

    all_movies_df=pd.DataFrame(all_movies_data,index=all_movies_id)

    all_movies_df.actor=all_movies_df.actor.apply(lambda x: actor_dict[x])
    all_movies_df.director=all_movies_df.director.apply(lambda x: director_dict[x])
    all_movies_df.rated=all_movies_df.rated.apply(lambda x: rated_dict[x])
    all_movies_df.genre=all_movies_df.genre.apply(lambda x: genre_dict[x])
    all_movies_df.year=all_movies_df.year - MOVIE_MIN_YEAR

    existing_movies_df=all_movies_df[all_movies_df.year<1998-MOVIE_MIN_YEAR]
    new_movies_df=all_movies_df[all_movies_df.year>1997-MOVIE_MIN_YEAR]

    #user_mask=np.random.rand(len(all_users_df)) < 0.8
    #user_existing=all_users_df[user_mask]
    #user_new=all_users_df[~user_mask]
    user_existing=all_users_df[all_users_df.index.isin(actual_users_index)]
    user_new=all_users_df[~all_users_df.index.isin(actual_users_index)]

    #To do : test exsiting movie with new user

    #To do : test new movie with existing user

    #To do : test new movie with new user



def get_movie_dict(movie_dict_file):
    with open(movie_dict_file,'r') as f:
        movie_dict=json.load(f)
    actor_dict=dict(zip(movie_dict['actors'],range(len(movie_dict['actors']))))
    director_dict=dict(zip(movie_dict['directors'],range(len(movie_dict['directors']))))
    rated_dict=dict(zip(movie_dict['rateds'],range(len(movie_dict['rateds']))))
    genre_dict=dict(zip(movie_dict['genres'],range(len(movie_dict['genres']))))
    return actor_dict,director_dict,rated_dict,genre_dict

def get_book_dict(book_dict_file):
    with open(book_dict_file,'r') as f:
        book_dict=json.load(f)
    author_dict=dict(zip(book_dict['authors'],range(len(book_dict['authors']))))
    publisher_dict=dict(zip(book_dict['publishers'],range(len(book_dict['publishers']))))
    return author_dict,publisher_dict

def ndcg(label,pred):
    pass

def dcg():
    pass

if __name__=='__main__':
    main()