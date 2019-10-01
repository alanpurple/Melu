import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import metrics,Model,layers,Sequential,losses,optimiziers
from tensorflow.keras.experimental import WideDeepModel,LinearModel
from connect_db import Session,engine
from data_models import Movie,User,Rating
import json
from model import MeluGlobal,MeluLocal
from sqlalchemy import func
from math import floor

MOVIE_MIN_YEAR=1919
MOVIE_MAX_YEAR=2000
MAX_USER_ID=6040

def main():
    linear_model=LinearModel()

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

if __name__=='__main__':
    main()