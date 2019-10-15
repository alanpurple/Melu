from tensorflow.keras import Model,layers,activations,Input
import tensorflow as tf

'''
class MeluLocal(Model):

    def __init__(self,layer_sizes):
        super(MeluLocal,self).__init__(name='MeluLocal')
        self.dense_layers=[]
        for size in layer_sizes:
            self.dense_layers.append(layers.Dense(size,activations.relu))
        self.final_layer=layers.Dense(1,activations.sigmoid)

    def call(self,inputs):
        x=inputs
        for layer in self.dense_layers:
            x=layer(x)
        return self.final_layer(x)
'''
def MeluLocal(emb_input_size,layer_sizes):
    local_input=Input((emb_input_size,))
    x=local_input
    for size in layer_sizes:
        x=layers.Dense(size,activations.relu)(x)
    output=layers.Dense(1,activations.sigmoid)(x)
    return Model(local_input,output)

    
'''
class MeluGlobal(Model):
    # 0 for bookcross, 1 for movielens
    def __init__(self,dict_sizes,emb_sizes,type=1):
        super(MeluGlobal,self).__init__(name='MeluGlobal')
        if type!=0 and type!=1:
            raise TypeError('wrong data type')
        self.type=type
        self.authdir_emb=layers.Embedding(dict_sizes['authdir'],emb_sizes['authdir'])
        self.year_emb=layers.Embedding(dict_sizes['year'],emb_sizes['year'])
        self.age_emb=layers.Embedding(dict_sizes['age'],emb_sizes['age'])
        self.concat=layers.Concatenate()
        if type==0:
            pass
        else:
            self.actor_emb=layers.Embedding(dict_sizes['actor'],emb_sizes['actor'])
            self.rated_emb=layers.Embedding(dict_sizes['rated'],emb_sizes['rated'])
            self.genre_emb=layers.Embedding(dict_sizes['genre'],emb_sizes['genre'])
            self.occu_emb=layers.Embedding(dict_sizes['occu'],emb_sizes['occu'])
            self.zipcode_emb=layers.Embedding(dict_sizes['zipcode'],emb_sizes['zipcode'])
            
    def call(self,inputs):
        authdir=self.authdir_emb(inputs['authdir'])
        year=self.year_emb(inputs['year'])
        age=self.age_emb(inputs['age'])
        if self.type==0:
            return
        else:
            actor=self.actor_emb(inputs['actor'])
            rated=self.rated_emb(inputs['rated'])
            genre=self.genre_emb(inputs['genre'])
            occu=self.occu_emb(inputs['occu'])
            zipicode=self.zipcode_emb(inputs['zipcode'])
            return self.concat([authdir,year,age,actor,rated,genre,occu])
'''
def MeluGlobal(dict_sizes,emb_sizes,type=1):
    if type!=0 and type!=1:
        raise TypeError('wrong data type')
    
    if type==0:
        # [authdir,year,age,pub]
        input=Input((4),name='book_input',dtype=tf.int32)
    else:
        # [authdir,year,age,actor,rated,genre,occu,zipcode]
        input=Input((8),name='movie_input',dtype=tf.int32)
    embeddings=[
        layers.Embedding(dict_sizes['authdir'],emb_sizes['authdir'],name='auth_emb')(input[:,0]),
        layers.Embedding(dict_sizes['year'],emb_sizes['year'],name='year_emb')(input[:,1]),
        layers.Embedding(dict_sizes['age'],emb_sizes['age'],name='age_emb')(input[:,2])
        ]
    if type==0:
        embeddings.append(layers.Embedding(dict_sizes['pub'],emb_sizes['pub'],name='pub_emb')(input[:,3]))
    else:
        embeddings.extend([
            layers.Embedding(dict_sizes['actor'],emb_sizes['actor'])(input[:,3]),
            layers.Embedding(dict_sizes['rated'],emb_sizes['rated'])(input[:,4]),
            layers.Embedding(dict_sizes['genre'],emb_sizes['genre'])(input[:,5]),
            layers.Embedding(dict_sizes['occu'],emb_sizes['occu'])(input[:,6]),
            layers.Embedding(dict_sizes['zipcode'],emb_sizes['zipcode'])(input[:,7])
            ])
    output=layers.Concatenate()(embeddings)

    return Model(input,output)