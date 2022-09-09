import os 
import time
from mido import MidiFile
from math import sqrt
import numpy as np
import tensorflow as tf
from .setting import seq_len, gen_len
from .Transformer import TransformerGenerator, TransformerDiscriminator

############## preprocessing ################

paths = []
songs = []

# walk through every file in target folder including itself
# edit your own data path
for r,d,f in os.walk('../data/magenta'):
    '''
    r: route
    d: folder in this folder
    f: file in this folder
    '''

    # if the file is mid file then append its path to paths[]
    for file in f:
        if '.mid' in file:
            paths.append(os.path.join(r,file))

# for each path in the array, create a Mido object and append it to songs
for path in paths:
    mid = MidiFile(path,type=1)
    songs.append(mid)


notes = []
dataset = []
chunk = []

# for each in midi object in list of songs
for i in range(len(songs)):
    for msg in songs[i]:
        # filtering out meta messages
        if not msg.is_meta:
            # filtering out control changes
            if (msg.type =='note_on'):
                notes.append(msg.note)

    for j in range(1,len(notes)):
        chunk.append(notes[j])
        # save each 16 note chunk
        if (j% seq_len==0):
            dataset.append(chunk)
            chunk = []
    print(f"Processing {i} Song")
    chunk=[]
    notes=[]

train_data = np.array(dataset)
np.save('preprocess_data.npz',train_data)



################ Training ################

# 定義 loss

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output),fake_output)

def discriminator_loss(real_output,fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output),real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output),fake_output)
    total_loss = real_loss + fake_loss

    return total_loss

# Train Step
BATCH_SIZE = 100
@tf.function
def train_step(music):
    LAMBDA = 10
    noise = tf.random.normal([BATCH_SIZE,seq_len])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_music = TransformerGenerator(hparams=noise,input_shape=[0,2,1,3],training=True)
        real_output = TransformerDiscriminator(hparams=music,inputs_shape=[0,2,1,3],training=True)
        fake_output = TransformerDiscriminator(generated_music,inputs_shape=[0,2,1,3],training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(fake_output,real_output)

    gradient_of_generator = gen_tape.gradient(gen_loss,TransformerGenerator.trainable_variables)
    gradient_of_discriminator = disc_tape.gradient(disc_loss,TransformerDiscriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradient_of_generator,TransformerGenerator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradient_of_discriminator,TransformerDiscriminator.trainable_variables))

    return gen_loss,disc_loss

# Train Loop 
import time
total_Gloss = []
total_Dloss = []
def train(dataset,epochs):
    for epoch in range(epochs):
        start = time.time()
        G_loss = 0
        D_loss = 0
        for i, image_batch in enumerate(dataset):
            gen_loss,disc_loss = train_step(image_batch)
            print(f"Step:{i} | G_loss: {gen_loss} D_loss: {disc_loss} |")
            G_loss += gen_loss
            D_loss += disc_loss

        clear_output(wait=True)
        print(f"Time for epoch {epoch +1} is {time.time()-start} sec\n")
        print(f"G_AVE_Loss: {G_loss/len(dataset)}")
        print(f"D_AVE_Loss: {D_loss/len(dataset)}")
        total_Gloss.append(G_loss/len(dataset))
        total_Dloss.append(D_loss/len(dataset))
