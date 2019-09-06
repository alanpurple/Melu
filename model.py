from tensorflow.keras import metrics,Model,layers,Sequential,activations,regularizers,Input

class MeluLocal(Model):

    def __init__(self):
        super(MeluLocal,self).__init__(name='MeluLocal')
        self.concat=layers.Concatenate()
        self.dense1=layers.Dense(64,activations.relu)
        self.dense2=layers.Dense(32,activations.relu)
        self.dense3=layers.Dense(16,activations.sigmoid)

    def call(self,inputs):
        x=self.concat(inputs)
        x=self.dense1(x)
        x=self.dense2(x)
        x=self.dense3(x)

class MeluGlobal(Model):

    def __init__(self,type=0):
        super(MeluGlobal,self).__init__(name='MeluGlobal')
        if type!=0 and type!=1:
            raise TypeError('wrong data type')
        else:
            self.type=type
            self.authdirec_emb=layers.Embedding()
            self.year_emb=layers.Embedding()
            if type==1:
                self.