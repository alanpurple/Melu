from sqlalchemy import Column,Integer,String,CHAR
from connect_db import base

class User(base):
    __tablename__='user'

    id=Column(Integer,primary_key=True)
    gender=Column(CHAR)
    age=Column(Integer)
    occupation=Column(Integer)
    zipcode=Column(Integer)