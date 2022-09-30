
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output
from .Params_setting import *

class GAN():
    def __init__(self,generator,discriminator,datapath,g_loss,d_loss,g_opt,d_opt,data_length=0):
        self.generator = generator
        self.discriminator = discriminator
        self.g_loss = g_loss
        self.d_loss = d_loss
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.train_data = np.load(datapath)
        self.gloss_list=[]
        self.dloss_list=[]
        self.gloss_name=''
        self.dloss_name=''

        if data_length > 0:
            self.train_data = self.train_data[:data_length]
        # rescale 0 to 1
        self.train_data = self.train_data.reshape(-1,256,1)
        self.train_dataset = tf.data.Dataset.from_tensor_slices(self.train_data).batch(Batch,drop_remainder=True)

    #### Train ####
    @tf.function
    def train_step(self,music):
        tf.random.set_seed(5)
        noise = tf.random.normal(shape=[Batch,Time,1],dtype=tf.float32,seed=12)
        # noise = tf.random.normal([Batch,Time,1])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_music = self.generator(noise,training=True)
            real_output = self.discriminator(music,training=True)
            fake_output = self.discriminator(generated_music,training=True)

            gen_loss = self.g_loss(fake_output)
            disc_loss = self.d_loss(real_output,fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss,self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss,self.discriminator.trainable_variables)
        self.g_opt.apply_gradients(zip(gradients_of_generator,self.generator.trainable_variables))
        self.d_opt.apply_gradients(zip(gradients_of_discriminator,self.discriminator.trainable_variables))
        
        return gen_loss,disc_loss



    def train(self,epochs):
        total_step = 0
        G_loss = 0
        D_loss = 0

        for epoch in range(epochs):
            start = time.time()

            for i, image_batch in enumerate(self.train_dataset):
                gen_loss, disc_loss = self.train_step(image_batch)
                print(f"Step: {i} || G_loss: {gen_loss}, D_loss: {disc_loss} ||")
                G_loss += gen_loss
                D_loss += disc_loss
                total_step += 1

                if total_step%100 == 0:
                    clear_output(wait=True)
                    print(f"G_AVE_loss: {G_loss/100}")
                    print(f"D_AVE_loss: {D_loss/100}")
                    self.gloss_list.append(G_loss/100)
                    self.dloss_list.append(D_loss/100)
                    G_loss = 0
                    D_loss = 0

            print(f"Time for epoch {epoch + 1} is {time.time()-start} sec\n")
        self.generator.save('./g_model')
        self.discriminator.save('./d_model')

    def predict(self,random_seed):
        tf.random.set_seed(5)
        noise = tf.random.normal(shape=[Batch,Time,1],dtype=tf.float32,seed=random_seed)  
        predict = self.generator.predict(noise)
        outputs = predict*128
        print(outputs)
        return outputs

    def save_loss(self,gloss_name,dloss_name):

        np.save(gloss_name,np.array(self.gloss_list))
        np.save(dloss_name,np.array(self.dloss_list))
        self.gloss_name = gloss_name
        self.dloss_name = dloss_name

    def show_graph(self):
        g= np.load(self.gloss_name+'.npy')
        month = [i for i in range(len(g))]
        plt.plot(g,'s-',color='r', label= 'generator loss')

        d = np.load(self.dloss_name+'.npy')
        plt.plot(d,'s-',color='b', label='discriminator loss')
        plt.legend(loc='best',fontsize=12)
        plt.show()

    