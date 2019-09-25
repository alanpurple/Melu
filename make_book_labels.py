import pandas as pd
import numpy as np
import re
import json

books_df=pd.read_csv('data/BX-Books.csv','\";\"',index_col=0,engine='python')

cols=books_df.columns.tolist()

cols[-1]='Image-URL-L'

books_df.columns=cols

books_df.index=[elem[1:] for elem in books_df.index]

books_df['Image-URL-L']=[elem[:-1] for elem in books_df['Image-URL-L']]

books_df.index.name='ISBN'

author_set=set(books_df['Book-Author'])
author_set.remove(np.nan)
authors=list(author_set)

pub_set=set(books_df.Publisher)
pub_set.remove(np.nan)
publishers=list(pub_set)

book_dict={
    'authors':authors,
    'publishers':publishers
}

with open('book_dict.json','w') as f:
    json.dump(book_dict,f)