from tensorflow.keras import Model,layers,activations

class MeluLocal(Model):

    def __init__(self,layer_sizes,num_layers=4):
        super(MeluLocal,self).__init__(name='MeluLocal')
        assert(len(layer_sizes)==num_layers)
        self.dense_layers=[]
        for i in range(num_layers):
            self.dense_layers.append(layers.Dense(layer_sizes[i],activations.relu))
        self.final_layer=layers.Dense(1,activations.sigmoid)

    def call(self,inputs):
        x=inputs
        for layer in self.dense_layers:
            x=layer(x)
        return self.final_layer(x)

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
            return self.concat([authdir,year,age,actor,rated,genre,occu])