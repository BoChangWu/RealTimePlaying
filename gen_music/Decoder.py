import tensorflow as tf
from tensorflow.keras.layers import Embedding,Dropout 
from DecoderLayer import DecoderLayer
from PositionalEncoding import positional_encoding

class Decoder(tf.keras.layers.Layer):
    # 初始參數跟 Encoder相同, 只差在用 'target_vocab_size' 而非 'inp_vocab_size'
    def __init__(self,num_layers,d_model,num_heads,dff,target_vocab_size,rate=0.1):
        super(Decoder,self).__init__()

        self.d_model = d_model

        # 為(目標語言) 建立詞嵌入層
        self.embedding = Embedding(target_vocab_size,d_model)
        self.pos_encoding = positional_encoding(target_vocab_size,self.d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads,dff,rate) for _ in range(num_layers)]
        self.dropout = Dropout(rate)

    # 呼叫時的參數跟 DecoderLayer一樣
    def call(self,x,enc_output,training,combined_mask,inp_padding_mask):

        tar_seq_len = tf.shape(x)[1]
        attention_weights={} # 用來存放每個 DecoderLayer 的注意權重

        # 跟 Encoder 做的事情完全一樣
        x= self.embedding(x) #(batch_size,tar_seq_len,d_model)
        x*= tf.math.sqrt(tf.cast(self.d_model,tf.float32))
        x+= self.pos_encoding[:,:tar_seq_len,:]
        x= self.dropout(x,training=training)

        for i, dec_layer in enumerate(self.dec_layers):
            x, block1,block2 = dec_layer(x,enc_output,training,combined_mask,inp_padding_mask)

            # 將從每個 DecoderLayer 取得的注意權重全部存下來回傳, 方便觀察
            attention_weights[f"decoder_layer{i+1}_block1"] = block1
            attention_weights[f"decoder_layer{i+1}_block2"] = block2

        # x.shape = (batch_size,tar_seq_len,d_model)
        return x,attention_weights