import time
import random
from IPython.display import clear_output
from mido import MidiFile, MidiTrack, Message
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from Transformer import TransformerGenerator,TransformerDiscriminator

#### Params ####
IntervalDim = 100
VelocityDim = 32
VelocityOffset = IntervalDim

NoteOnDim = NoteOffDim = 128
NoteOnOffset = IntervalDim + VelocityDim
NoteOffOffset = IntervalDim + VelocityDim + NoteOnDim

CCDim = 2
CCOffset = IntervalDim + VelocityDim + NoteOnDim + NoteOffDim
EventDim = IntervalDim + VelocityDim + NoteOnDim + NoteOffDim + CCDim # 390

Time = 256
EmbeddingDim = 512
HeadDim = 16
Heads = 16
ContextDim = HeadDim * Heads # 512
Layers = 8

def default_hparams():
    return {
        'EventDim': EventDim,
        "ContextDim": ContextDim,
        'EmbeddingDim': EmbeddingDim,
        'Heads': Heads,
        'Layers': Layers,
        'Time': Time
    }

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
train_data = train_data[:1000]
# rescale 0 to 1
train_data = train_data.reshape(-1,256,1)
#########################
Batch = 1

#### Optimizer ####
generator_optimizer = Adam(1e-4)
discriminator_optimizer = Adam(1e-4)
###################

#### Loss function ####
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output),fake_output)

def discriminator_loss(real_output,fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output),real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output),fake_output)
    total_loss = real_loss + fake_loss

    return total_loss
#######################

#### Train ####
@tf.function
def train_step(music):
    noise = tf.random.normal([Batch,Time,1])

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
                print(f"D_AVE_loss: {G_loss/100}")
                gloss_list.append(G_loss/100)
                dloss_list.append(D_loss/100)
                G_loss = 0
                D_loss = 0

        print(f"Time for epoch {epoch + 1} is {time.time()-start} sec\n")
###############

#### Main ####
train_dataset = tf.data.Dataset.from_tensor_slices(train_data).batch(Batch,drop_remainder=True)

train(train_dataset,1)

np.save('0912_MTgan_gloss',np.array(gloss_list))
np.save('0912_MTgan_dloss',np.array(dloss_list))

#############

#### Make Midi ####
noise = tf.random.normal([Batch,Time,1])
predict = generator.predict(noise)
predict = predict*128
print(predict)

midler = MidiFile()
track = MidiTrack()

midler.tracks.append(track)
track.append(Message('program_change',program=2,time=0))

for x in range(256):
    on_interval = random.randint(50,127)
    off_interval = random.randint(0,127)
    change_interval = random.randint(0,127)
    change_value = random.randint(0,127)
    isControl = random.randint(0,1)
    
    track.append(Message('note_on', channel=1, note=int(predict[0][x][0]), velocity=64, time=on_interval))

    if isControl:
        track.append(Message('control_change', channel=1, control=64, value=change_value, time= change_interval))

    track.append(Message('note_off', channel=1, note=int(predict[0][x][0]), velocity=64, time=off_interval))

midler.save('MT_song.mid')
###################

#### show graph ####
g= np.load('0912_MTgan_gloss.npy')
month = [i for i in range(len(g))]
plt.plot(g,'s-',color='r', label= 'generator loss')

d = np.load('0912_MTgan_dloss.npy')
plt.plot(d,'s-',color='b', label='discriminator loss')
plt.legend(loc='best',fontsize=12)
plt.show()


