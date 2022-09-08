import tensorflow as tf
from tensorflow.keras.layers import (Dense,LayerNormalization,Dropout)
from .Feed_Forward_Network import feed_forward_network
from .MultiHeadAttention import do_MultiHeadAttention


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,d_model,num_heads,dff,rate=0.1):
        super(DecoderLayer,self).__init__()

        self.ffn = feed_forward_network(d_model,dff)

        # 定義每個 sub-layer 用的 LayerNorm
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)

        # 定義每個 sub-layer 用的 Dropout
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)

    
    def call(self,x,enc_output,training,combined_mask,inp_padding_mask):

        # enc_output 為 Encoder 輸出序列, shape為 (batch_size,input_seq_len,d_model)
        # attn_weight_block_1 則為 (batch_size, num_heads, target_seq_len, target_seq_len)
        # attn_weight_block_2 則為 (batch_size, num_heads, target_seq_len, input_seq_len)
        # 同時需要 look ahead mask 以及輸出序列的 padding mask

        attn1,attn_weight_block1 = do_MultiHeadAttention(x,x,x,combined_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1+x)

        # sub-layer 2: DecoderLayer 關注 Encoder 的最後輸出
        # 一樣需要對 Encoder 的輸出套用 padding mask
        attn2, attn_weight_block2 = do_MultiHeadAttention(enc_output,enc_output,out1,inp_padding_mask)

        #(batch_size,target_seq,len,d_model)
        attn2 = self.dropout2(attn2,training=training)
        out2 = self.layernorm2(attn2+out1)

        # sub-layer 3: FFN 部分跟 EncdoerLayer一樣
        ffn_output = self.ffn(out2) # (batch_size,target_seq_len,d_model)
        ffn_output = self.dropout3(ffn_output,training=training)
        out3 = self.layernorm3(ffn_output+out2) #(batch_size,target_seq_len,d_model)

        # 除了主要輸出 'out3' 以外, 輸出 multi-head 注意權重方便之後理解模型內部狀況
        return out3,attn_weight_block1,attn_weight_block2
