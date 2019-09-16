from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from urllib import parse

server='10.102.40.94'
database='movielens'
driver='MySQL ODBC 8.0 Unicode Driver'
id='root'
pwd='wmind'


engine=create_engine('mysql+pyodbc://{}:{}@{}/{}?driver={}'.format(id,pwd,server,database,parse.quote_plus(driver)))
base=declarative_base()