from sqlalchemy import Column,Integer,String,CHAR
from connect_db import base

class Movie(base):
    __tablename__='movie'

    id=Column(Integer,primary_key=True)
    title=Column(String)
    year=Column(Integer)
    actor=Column(String)
    director=Column(String)
    rated=Column(CHAR)