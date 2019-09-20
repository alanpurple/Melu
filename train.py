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



if __name__=='__main__':
    main()