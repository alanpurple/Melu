from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column,Integer,String,CHAR,ForeignKey
from sqlalchemy.orm import relationship


Base=declarative_base()

class Rating(Base):
    id=Column(Integer,primary_key=True)
    user_id=Column(Integer,ForeignKey('users.id'))
    movie_id=Column(Integer,ForeignKey('movies.id'))
    rate=Column(Integer)