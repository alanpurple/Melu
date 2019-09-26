import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import metrics,Model,layers,Sequential,losses
from connect_db import Session,engine
from data_models import Movie,User,Rating
import json
from model import MeluGlobal,MeluLocal
from sqlalchemy import func

# get all rating
# divide movies before 1997 and after 1998 ( approximately 8:2 )

# divide user into new and existing group
# remove rating for existing items rated by new users
# remove rating for new items rated by existing users

MOVIE_MIN_YEAR=1919
MOVIE_MAX_YEAR=2000
MAX_USER_ID=6040

def main():
    session=Session()
    # query with condition? alternative
    all_users=session.query(User).all()
    all_movies=session.query(Movie).all()

    actor_dict,director_dict,rated_dict,genre_dict=get_movie_dict('movie_dict.json')
    #author_dict,publisher_dict=get_book_dict('book_dict.json')
    
    with open('movie_user_zipcodes.json','r') as f:
        zipcodes=json.load(f)
    zipcode_dict=dict(zip(zipcodes,range(len(zipcodes))))

    all_users_id=[elem.id for elem in all_users]
    all_users_data=[{'gender':elem.gender,'occupation':elem.occupation,'age':elem.age,'zipcode':elem.zipcode} for elem in all_users]

    all_users_df=pd.DataFrame(all_users_data,index=all_users_id)

    all_users_df.gender=(all_users_df.gender=='M').astype(int)
    all_users_df.zipcode=all_users_df.zipcode.apply(lambda x: zipcode_dict[x])

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

    user_mask=np.random.rand(len(all_users_df)) < 0.8
    user_existing=all_users_df[user_mask]
    user_new=all_users_df[~user_mask]

    rating_existing=session.query(Rating).join(User).filter(User.id.in_(user_existing.index)).join(Movie).filter(Movie.year<1998).all()
    rating_exist_new=session.query(Rating).join(User).filter(User.id.in_(user_existing.index)).join(Movie).filter(Movie.year>1997).all()
    rating_new_exist=session.query(Rating).join(User).filter(User.id.in_(user_new.index)).join(Movie).filter(Movie.year<1998).all()
    rating_new_new=session.query(Rating).join(User).filter(User.id.in_(user_new.index)).join(Movie).filter(Movie.year>1997).all()

    train_genders=[1 if elem.user.genre=='M' else 0 for elem in rating_existing]
    train_occupations=[elem.user.occupation for elem in rating_existing]
    train_ages=[elem.user.age for elem in rating_existing]
    train_zipcodes=[all_users_df.loc[elem.user_id].zipcode for elem in rating_existing]
    train_actors=[all_movies_df.loc[elem.movie_id].actor for elem in rating_existing]
    train_directors=[all_movies_df.loc[elem.movie_id].director for elem in rating_existing]
    train_genres=[all_movies_df.loc[elem.movie_id].genre for elem in rating_existing]
    train_rateds=[all_movies_df.loc[elem.movie_id].rated for elem in rating_existing]

    train_labels=[(elem.rate-1)*0.25 for elem in rating_existing]

    rating_existing_group=[[] for _ in range(MAX_USER_ID+1)]
    for rating in rating_existing:
        rating_existing_group[rating.user_id].append(rating)


    dict_sizes={'zipcode':len(zipcode_dict),'actor':len(actor_dict),
                'authdir':len(director_dict),'rated':len(rated_dict),
                'year':MOVIE_MAX_YEAR-MOVIE_MIN_YEAR+1}
    emb_sizes={'zipcode':100,'actor':50,'authdir':50,'rated':5,'year':15}

    local_model=MeluLocal([64,32,16,4],4)
    global_model=MeluGlobal(dict_sizes,emb_sizes,1)
    
    for epoch in range(30):
        with tf.GradientTape() as tape:
            pass


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

def extract_movie_data(data):
    pass

if __name__=='__main__':
    main()