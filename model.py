from tensorflow.keras import metrics,Model,layers,Sequential,activations,regularizers,Input

class MeluLocal(Model):

    def __init__(self,layer_sizes,num_layers=5):
        super(MeluLocal,self).__init__(name='MeluLocal')
        assert(len(layer_sizes)==num_layers)
        self.layers=[]
        for i in range(num_layers-1):
            self.layers.append(layers.Dense(layer_sizes[i]),activations.relu)
        self.final_layer=layers.Dense(layer_sizes[-1],activations.sigmoid)

    def call(self,inputs):
        x=inputs
        for layer in self.layers:
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
        self.concat=layers.Concatenate()
        if type==0:
            pass
        else:
            self.actor_emb=layers.Embedding(dict_sizes['actor'],emb_sizes['actor'])
            self.rated_emb=layers.Embedding(dict_sizes['rated'],emb_sizes['rated'])
            self.genre_emb=layers.Embedding(dict_sizes['genre'],emb_sizes['genre'])
            
    def call(self,inputs):
        authdir=self.authdir_emb(inputs['authdir'])
        year=self.year_emb(inputs['year'])
        if self.type==0:
            return
        else:
            actor=self.actor_emb(inputs['actor'])
            rated=self.rated_emb(inputs['rated'])
            genre=self.genre_emb(inputs['genre'])
            return self.concat([authdir,year,actor,rated,genre])