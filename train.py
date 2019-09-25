import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import metrics,Model,layers,Sequential,losses
from connect_db import Session,engine
from data_models import Movie,User,Rating

# get all rating
# divide movies before 1997 and after 1998 ( approximately 8:2 )

# divide user into new and existing group
# remove rating for existing items rated by new users
# remove rating for new items rated by existing users


def main():
    session=Session()
    # query with condition? alternative
    all_users=session.query(User).all()
    existing_movies=session.query(Movie).filter(Movie.year<1998).all()
    new_movies=session.query(Movie).filter(Movie.year>1997).all()

    all_users_id=[elem.id for elem in all_users]
    all_users_data=[{'gender':elem.gender,'occupation':elem.occupation,'age':elem.age,'zipcode':elem.zipcode} for elem in all_users]

    all_users_df=pd.DataFrame(all_users_data,index=all_users_id)

    existing_movies_id=[elem.id for elem in existing_movies]
    existing_movies_data=[{
        'year':elem.year,'actor':elem.actor,'title':elem.title,'rated':elem.rated,
        'director':elem.director,'genre':elem.genre
        } for elem in existing_movies]
    new_movies_id=[elem.id for elem in new_movies]
    new_movies_data=[{
        'year':elem.year,'actor':elem.actor,'title':elem.title,'rated':elem.rated,
        'director':elem.director,'genre':elem.genre
        } for elem in new_movies]

    existing_movies_df=pd.DataFrame(existing_movies_data,index=existing_movies_id)
    new_movies_df=pd.DataFrame(new_movies_data,index=new_movies_id)

    user_mask=np.random.rand(len(all_users_df)) < 0.8
    user_existing=all_users_df[user_mask]
    user_new=all_users_df[~user_mask]

    rating_existing=session.query(Rating).join(User).filter(User.id.in_(user_existing.index)).join(Movie).filter(Movie.year<1998).all()
    rating_exist_new=session.query(Rating).join(User).filter(User.id.in_(user_existing.index)).join(Movie).filter(Movie.year>1997).all()
    rating_new_exist=session.query(Rating).join(User).filter(User.id.in_(user_new.index)).join(Movie).filter(Movie.year<1998).all()
    rating_new_new=session.query(Rating).join(User).filter(User.id.in_(user_new.index)).join(Movie).filter(Movie.year>1997).all()

def extract_movie_data(data):
    pass

if __name__=='__main__':
    main()