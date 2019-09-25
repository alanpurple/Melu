from connect_db import Session,engine
from data_models import Movie,User,Rating
import json

session=Session()

movies=session.query(Movie.director,Movie.actor,Movie.rated,Movie.genre).all()
books=session.query()

directors=list({elem.director for elem in movies})
actors=list({elem.actor for elem in movies})
rateds=list({elem.rated for elem in movies})
genres=list({elem.genre for elem in movies})

movie_dict={
    'directors':directors,
    'actors':actors,
    'rateds':rateds,
    'genres':genres
}

with open('movie_dict.json','w') as f:
    json.dump(movie_dict,f)