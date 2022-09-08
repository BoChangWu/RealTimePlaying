import tensorflow as tf
from tensorflow.keras.layers import Dense
def feed_forward_network(d_model,dff):
    # 此 FFN 輸入做兩個線性轉換, 中間還加了 ReLU

    return tensorflow.kerase.Sequential([
        Dense(dff,activation='relu'), #(batch_size,seq_len,dff)
        Dense(d_model) #(batch_size,seq_len,d_model)
    ])