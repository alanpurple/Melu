import tensorflow as tf
import pandas as pd
import numpy as np
from connect_db import Session,engine
from data_models import User
import json

session=Session()
all_users=session.query(User.zipcode).all()

zipcodes=list(set([elem[0] for elem in all_users]))

with open('movie_user_zipcodes.json','w') as f:
    json.dump(zipcodes,f)