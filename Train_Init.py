import tensorflow as tf

from Transformer import TransformerGenerator,TransformerDiscriminator
from .Optimize import generator_optimizer, discriminator_optimizer,generator_loss, discriminator_loss
from .Midi_Edit import Midi_Encode
from .GAN_Module import GAN
#### Params ####
from .Params_setting import *

hparams = default_hparams()

print('hparams["Heads"]: ',hparams['Heads'])
################

#### Models ####
generator = TransformerGenerator(hparams,input_shape=(256,1))
generator.summary()

discriminator = TransformerDiscriminator(hparams, input_shape=(256,1))
discriminator.summary()
###############

#### Train ####
Gan = GAN(generator= generator,
                discriminator= discriminator,
                datapath= 'preprocess_data.npy',
                g_loss= generator_loss,
                d_loss= discriminator_loss,
                g_opt= generator_optimizer,
                d_opt= discriminator_optimizer,
                data_length=10000)
with tf.device('/device:GPU:0'):
    Gan.train(epochs=1)

Gan.save_loss(gloss_name='0912_MTgan_gloss',dloss_name='0912_MTgan_dloss')

#### predict ####
predict = Gan.predict()

#### Generate Midifile ####
midi_encode = Midi_Encode()

midi_encode.make_file(params=predict,name='Song#1.mid')

#### show graph ####
Gan.show_graph()


