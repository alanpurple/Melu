from sqlalchemy import Column,Integer,String,CHAR,ForeignKey
from sqlalchemy.orm import relationship
from connect_db import base

class Rating(base):

    __tablename__='rating'

    id=Column(Integer,primary_key=True)
    user_id=Column(Integer,ForeignKey('user.id'))
    movie_id=Column(Integer,ForeignKey('movie.id'))
    rate=Column(Integer)