import time
import os
import random
from IPython.display import clear_output
from mido import MidiFile, MidiTrack, Message
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model


from Transformer import TransformerGenerator,TransformerDiscriminator
from .Optimize import generator_optimizer, discriminator_optimizer,generator_loss, discriminator_loss
from .Midi_Edit import Midi_Encode
#### Params ####
from .Pparams_setting import *

hparams = default_hparams()

print('hparams["Heads"]: ',hparams['Heads'])
################

#### Models ####
generator = TransformerGenerator(hparams,input_shape=(256,1))
generator.summary()

discriminator = TransformerDiscriminator(hparams, input_shape=(256,1))
discriminator.summary()
################

#### preprocess data ####
train_data = np.load('preprocess_data.npy')
# train_data = train_data[:60000]
# rescale 0 to 1
train_data = train_data.reshape(-1,256,1)
#########################

#### Train ####
@tf.function
def train_step(music):
    tf.random.set_seed(5)
    noise = tf.random.normal(shape=[Batch,Time,1],dtype=tf.float32,seed=12)
    # noise = tf.random.normal([Batch,Time,1])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_music = generator(noise,training=True)
        real_output = discriminator(music,training=True)
        fake_output = discriminator(generated_music,training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output,fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss,generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss,discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator,generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,discriminator.trainable_variables))
    
    return gen_loss,disc_loss

gloss_list=[]
dloss_list=[]

def train(dataset,epochs):
    total_step = 0
    G_loss = 0
    D_loss = 0

    for epoch in range(epochs):
        start = time.time()

        for i, image_batch in enumerate(dataset):
            gen_loss, disc_loss = train_step(image_batch)
            print(f"Step: {i} || G_loss: {gen_loss}, D_loss: {disc_loss} ||")
            G_loss += gen_loss
            D_loss += disc_loss
            total_step += 1

            if total_step%100 == 0:
                clear_output(wait=True)
                print(f"G_AVE_loss: {G_loss/100}")
                print(f"D_AVE_loss: {D_loss/100}")
                gloss_list.append(G_loss/100)
                dloss_list.append(D_loss/100)
                G_loss = 0
                D_loss = 0

        print(f"Time for epoch {epoch + 1} is {time.time()-start} sec\n")
    generator.save('./g_model')
    discriminator.save('./d_model')
###############


#### Main ####
train_dataset = tf.data.Dataset.from_tensor_slices(train_data).batch(Batch,drop_remainder=True)

with tf.device('/device:GPU:0'):
    train(train_dataset,1)
# gckpt.save(file_prefix=g_prefix)
# dckpt.save(file_prefix=d_prefix)

np.save('0912_MTgan_gloss',np.array(gloss_list))
np.save('0912_MTgan_dloss',np.array(dloss_list))

#############

#### predict ####
tf.random.set_seed(5)
noise = tf.random.normal(shape=[Batch,Time,1],dtype=tf.float32,seed=6)
predict = generator.predict(noise)
predict = predict*128
print(predict)
#################

#### Generate Midifile ####
midi_encode = Midi_Encode()

midi_encode.make_file(params=predict,name='Song#1.mid')
###########################

#### show graph ####
g= np.load('0912_MTgan_gloss.npy')
month = [i for i in range(len(g))]
plt.plot(g,'s-',color='r', label= 'generator loss')

d = np.load('0912_MTgan_dloss.npy')
plt.plot(d,'s-',color='b', label='discriminator loss')
plt.legend(loc='best',fontsize=12)
plt.show()


