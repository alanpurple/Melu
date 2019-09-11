from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column,Integer,String,CHAR

Base=declarative_base()

class Movie(Base):
    #__tablename__='movie'

    id=Column(Integer,primary_key=True)
    title=Column(String)
    year=Column(Integer)
    actor=Column(String)
    director=Column(String)
    rated=Column(CHAR)