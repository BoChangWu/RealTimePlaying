import time
import os
import random
from IPython.display import clear_output
from mido import MidiFile, MidiTrack, Message
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from Transformer import TransformerGenerator,TransformerDiscriminator
from .Pparams_setting import *
from .Midi_Edit import Midi_Encode
#### Params ####

hparams = default_hparams()

print('hparams["Heads"]: ',hparams['Heads'])
################

#### Models ####
generator = tf.keras.models.load_model('./g_model')
generator.summary()

discriminator = tf.keras.models.load_model('./d_model')
discriminator.summary()
################

for i  in range(10):
    tf.random.set_seed(10)
    noise = tf.random.normal(shape=[Batch,Time,1],dtype=tf.float32,seed=random.randint(0,127))
    print(noise)
    predict = generator.predict(noise)
    predict = predict*128
    print(predict)

    midi_encode = Midi_Encode()

    midi_encode.make_file(params=predict,name=f'./test/sample{i}.mid')