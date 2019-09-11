from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column,Integer,String,CHAR

Base=declarative_base()

class User(Base):

    id=Column(Integer,primary_key=True)
    gender=Column(CHAR)
    age=Column(Integer)
    occupation=Column(Integer)
    zipcode=Column(Integer)