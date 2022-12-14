import tensorflow as tf
from tensorflow import Dense
from .Encoder import Encoder
from .Decoder import Decoder

class Transformer(tf.keras.Model):
    def __init__(self,num_layers,d_model,num_heads,dff,input_vocab_size,target_vocab_size,rate=0.1):
        super(Transformer,self).__init__()

        self.encoder = Encoder(num_layers,d_model,num_heads,dff,input_vocab_size,rate)
        self.decoder = Decoder(num_layers,d_model,num_heads,dff,target_vocab_size,rate)

        # 這個 FFN 輸出一樣大的 logits 數, 等通過 softmax 就代表每個輸出的出現機率
        self.final_layer = Dense(target_vocab_size)

    
    def call(self,inp,tar,training,enc_padding_mask,combined_mask,dec_padding_mask):

        enc_output = self.encoder(inp,training,enc_padding_mask)
        dec_output,attention_weights = self.decoder(tar,enc_output,training,combined_mask,dec_padding_mask)
        final_output = self.final_layer(dec_output)

        return final_output,attention_weights