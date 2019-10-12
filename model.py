from tensorflow.keras import Model,layers,activations,Input

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
    assert(len(layer_sizes)==num_layers)
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
    authdir_input=Input((dict_sizes['authdir'],))
    year_input=Input((dict_sizes['year'],))
    age_input=Input((dict_sizes['age'],))
    inputs=[authdir_input,year_input,age_input]
    embeddings=[
        layers.Embedding(dict_sizes['authdir'],emb_sizes['authdir'])(authdiractor_input),
        layers.Embedding(dict_sizes['year'],emb_sizes['year'])(year_input),
        layers.Embedding(dict_sizes['age'],emb_sizes['age'])(age_input)
        ]
    if type==0:
        pass
    else:
        actor_input=Input((dict_sizes['actor'],))
        rated_input=Input((dict_sizes['rated'],))
        genre_input=Input((dict_sizes['genre'],))
        occu_input=Input((dict_sizes['occu'],))
        zipcode_input=Input((dict_sizes['zipcode'],))
        inputs.extend([aactor_input,rated_input,genre_input,occu_input,zipcode_input])
        embeddings.extend([
            layers.Embedding(dict_sizes['actor'],emb_sizes['actor'])(actor_inactor_input),
            layers.Embedding(dict_sizes['rated'],emb_sizes['rated'])(rated_input),
            layers.Embedding(dict_sizes['genre'],emb_sizes['genre'])(genre_input),
            layers.Embedding(dict_sizes['occu'],emb_sizes['occu'])(occu_input),
            layers.Embedding(dict_sizes['zipcode'],emb_sizes['zipcode'])(zipcode_input)
            ])
    output=layers.Concatenate()(embeddings)

    return Model(inputs,output)