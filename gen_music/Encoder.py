import tensorflow as tf
from tensorflow.keras.layers import Embedding,Dropout 
from EncoderLayer import EncoderLayer
from PositionalEncoding import positional_encoding

class Encoder(tf.keras.layers.Layer):
    '''
    
    Encoder 的初始參數本來就要給 EncoderLayer, 還多了:
    num_layers: 決定要有幾個 EncoderLayers
    input_vocab_size: 用來把索引轉成詞嵌入向量
    
    '''

    def __init__(self,num_layers,d_model,num_heads,dff,input_vocab_size,rate=0.1):
        super(Encoder,self).__init__()

        self.d_model = d_model # 這是長度不是model
        
        # Input 進來要先 embedding
        self.embedding = tf.keras.layers.Embedding(input_vocab_size,d_model)
        
        self.pos_encoding = positional_encoding(input_vocab_size,d_model)

        self.enc_layers = [EncoderLayer(d_model,num_heads,dff,rate) for _ in range(num_layers)]

        self.dropout = Dropout(rate)

    
    def call(self,x,training,mask):
        # 輸入的 x.shape == (batch_size,input_seq_len,d_model)
        # 以下各 layer 的輸出皆為 (batch_size, input_seq_len,d_model)
        input_seq_len = tf.shape(x)[1]

        # 將 二維的索引序列轉成 三維的詞嵌入張量, 並乘上 sqrt(d_model)
        # 再加上對應長度的 position emcoding

        x= self.embedding(x)
        x*= tf.math.sqrt(tf.cast(self.d_model,tf.float32))
        x+= self.pos_encoding[:,:input_seq_len,:]

        # 對 embedding 與 position encoding 的總和做 regularization
        # regularization 與 Dropout 用來預防 overfitting
        
        # 通過 N個 EncoderLayer做編碼
        for i,enc_layer in enumerate(self.enc_layers):
            x = enc_layer(x,training,mask)

        return x