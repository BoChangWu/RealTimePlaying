import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense 
from .setting import seq_len,num_head,depth
def do_attention(q,k,v,mask):

    # k矩陣轉置與 q 相乘
    matmul_qk = tf.linalg.matmul(q,k,tranpose_b=True)

    seq_k_length = tf.cast(tf.shape(k)[-1,tf.float32])
    scaled_attention_logits = matmul_qk/tf.math.sqrt(seq_k_length)

    # 將 mask 加到被丟入 softmax 前的 logits 
    # 這裡的 mask 可以是純 padding 或combined_mask

    if mask != None:
        scaled_attention_logits +=(mask*-1e9) # 讓被加上極大負值的位置變得無關緊要

    # 取 softmax 是為了得到總和為 1 的比例之後對 'v' 做加權平均
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    
    # 以注意權重對 v 做加權平均(weighted average)
    return tf.matmul(attention_weights,v)


def split_heads(x,batch_size,num_head,depth):
    x = tf.reshape(x,(batch_size,-1,num_head,depth))
    return tf.transpose(x,perm=[0,2,1,3])




def do_MultiHeadAttention(q,k,v,mask):
    batch_size = tf.shape(q)[0]

    # q,k,v 都會做一次線性轉換到 seq_len 維的空間
    q= Dense(seq_len)(q)
    q= split_heads(q,batch_size,num_head,depth)

    k= Dense(seq_len)(k)
    k= split_heads(k,batch_size,num_head,depth)

    v= Dense(seq_len)(v)
    v= split_heads(v,batch_size,num_head,depth)

    scaled_attention,attention_weight = do_attention(q,k,v,mask)

    # 把 multihead 結果接回
    scaled_attention = tf.transpose(scaled_attention,perm=[0,2,1,3])
    concat_attention = tf.reshape(scaled_attention,(batch_size,-1,seq_len))

    # 最後再通過一次線性轉換
    output = Dense(concat_attention)

    return output
