import tensorflow as tf
from tensorflow.keras.layers import (Dense,LayerNormalization,Dropout)
from .Feed_Forward_Network import feed_forward_network
from .MultiHeadAttention import do_MultiHeadAttention

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self,d_model,num_heads,dff,rate=0.1):
        super(EncoderLayer,self).__init__()

        self.ffn = feed_forward_network(d_model,dff)
        # layer norm 很常再 RNN-based 的模型被使用, 一個 sub-layer, 一個 layer norm
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        # 一個 sub-layer 一個 dropout layer
        # 預防 overfitting
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)


    # 丟入 'isTraining' 參數因為 dropout 在訓練以及測試的行為不同
    def call(self,x,mask,isTraining=True):
        # 除了 'attn', 其他tensor 的 shape 皆為(batch_size,input_seq_len,d_model)
        # attn.shape = (batch_size,num_head,input_seq_len,input_seq_len)
        # Encoder 利用self-attention機制, 因此 q,k,v 全都是自己
        # 還需要 padding mask 來mask 輸入序列
        attn_output,attn = do_MultiHeadAttention(x,x,x,mask)
        attn_output = self.dropout1(attn_output,training=isTraining)
        out1 = self.layernorm1(x+attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output,training=isTraining)
        out2 = self.layernorm2(out1+ffn_output)

        return out2