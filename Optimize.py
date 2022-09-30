import tensorflow as tf
from tensorflow.keras.optimizers import Adam


#### Loss Function ####
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output),fake_output)

def discriminator_loss(real_output,fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output),real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output),fake_output)
    total_loss = real_loss + fake_loss

    return total_loss

generator_optimizer = Adam(learning_rate=1e-9,beta_1=0.8,beta_2=0.999,amsgrad=True,epsilon=1e-8)
discriminator_optimizer = Adam(learning_rate=1e-9,beta_1=0.8,beta_2=0.999,amsgrad=True,epsilon=1e-8)