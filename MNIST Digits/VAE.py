import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from IPython import display
import matplotlib.pyplot as plt


# Preprocessing of data
def preprocess_images(images):
  images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
  return np.where(images > .50, 1.0, 0.0).astype('float32')


# Create model
class VAE(tf.keras.Model):

  def __init__(self, latent_space):
    super(VAE, self).__init__()
    self.latent_space = latent_space

    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='selu'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='selu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='selu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='selu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim*latent_dim, activation="relu"),
            tf.keras.layers.Dense(latent_space + latent_space),
        ]
    )

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_space,)),
            tf.keras.layers.Dense(latent_dim*latent_dim, activation="relu"),
            tf.keras.layers.Dense(units=7*7*64,),
            tf.keras.layers.Reshape(target_shape=(7, 7, 64)),      
            tf.keras.layers.UpSampling2D((2,2)),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, padding='same', activation='selu'),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, padding='same', activation='selu'),
            tf.keras.layers.UpSampling2D((2,2)),
            tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, padding='same', activation='selu'),
            tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, padding='same', activation='selu'),            
            tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, padding='same'),            
        ]
    )
    
  def encode(self, x):
    mean, var = tf.split(self.encoder(x), 
                         num_or_size_splits=2, axis=1)
    return mean, var

  def decode(self, z, sigmoid=False):
    pred = self.decoder(z)
    if sigmoid:
      probs = tf.sigmoid(pred)
      return probs
    return pred

  @tf.function
  def sample(self, noise=None):
    if noise is None:
      noise = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(noise, sigmoid=True)

  # Avoid the bottleneck of backpropagation
  def reparameterize(self, mean, var):
    norm = tf.random.normal(shape=mean.shape)
    return mean + tf.exp(var * .5)*norm



# ELBO
def norm_log(sample, mean, var, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-var) + var + log2pi),
      axis=raxis)

# ELBO 
def compute_loss(model, x):
  mean, var = model.encode(x)
  z = model.reparameterize(mean, var)
  decode_x = model.decode(z)
  sigmoid_cross = tf.nn.sigmoid_cross_entropy_with_logits(logits=decode_x, labels=x)
  log_x_z = -tf.reduce_sum(sigmoid_cross, axis=[1, 2, 3])
  log_z = norm_log(z, 0., 0.)
  log_z_x = norm_log(z, mean, var)
  return -tf.reduce_mean(log_x_z + log_z - log_z_x)


# Define the training step  
@tf.function
def train_step(model, x, optimizer):
    
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  

# Generate the images
def generate_and_save_images(model, epoch, test_sample):
  mean, var = model.encode(test_sample)
  z = model.reparameterize(mean, var)
  predictions = model.sample(z)
  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(predictions[i, :, :, 0], cmap='gray')
    plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()
    

# Hyperparameters
#tf.random.set_seed(0)

epochs = 30
latent_dim = 20
numb_to_generate = 16  

train_size = 1000
batch_size = 25
test_size = 1000

optimizer = tf.keras.optimizers.Adam(1e-3)


# Main Code
(train_images, label_x), (test_images, label_y) = tf.keras.datasets.mnist.load_data()

train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)
    
train_images = train_images[:1000]
test_images = test_images[:1000]

train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                 .shuffle(train_size).batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                .shuffle(test_size).batch(batch_size))


# Generate random image
generate_noise = tf.random.normal(shape=[numb_to_generate, latent_dim])

model = VAE(latent_dim)



assert batch_size >= numb_to_generate
for test_batch in test_dataset.take(1):
  test_sample = test_batch[0:numb_to_generate, :, :, :]

  
for epoch in range(1, epochs + 1):
  start_time = time.time()
  for train_x in train_dataset:
    train_step(model, train_x, optimizer)
  end_time = time.time()

  loss = tf.keras.metrics.Mean()
  for test_x in test_dataset:
    loss(compute_loss(model, test_x))
  elbo = -loss.result()
  display.clear_output(wait=False)
  print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
        .format(epoch, elbo, end_time - start_time))


  generate_and_save_images(model, epoch, test_sample)