from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

import sys
import cv2
import matplotlib.pyplot as plt

from keras.layers import Permute, Dense, Lambda, RepeatVector, merge
from keras.optimizers import Adam
from keras.layers import Input, Dense, Activation, BatchNormalization, UpSampling2D, Convolution2D, LeakyReLU, Flatten, Dropout, Reshape
from keras.models import Model

from tqdm import tqdm

from classes.dataset.Generator import *

def plot_gen(n_ex=16,dim=(4,4), figsize=(10,10) ):
    noise = np.random.uniform(0,1,size=[n_ex,100])
    generated_images = generator.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0],dim[1],i+1)
        img = generated_images[i,0,:,:]
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_loss(losses):
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.figure(figsize=(10,8))
        plt.plot(losses["d"], label='discriminitive loss')
        plt.plot(losses["g"], label='generative loss')
        plt.legend()
        plt.show()


input_path = "../datasets/web/training_features"
val_path = "../datasets/web/eval_set/"
output_path = "../bin"
use_generator = True
pretrained_weigths = None
np.random.seed(1234)
data_percentage = 1.0

dataset = Dataset()
dataset.load(input_path, generate_binary_sequences=True,
             data_percentage=data_percentage)
dataset.save_metadata(output_path)
dataset.voc.save(output_path)

gui_paths, img_paths = Dataset.load_paths_only(input_path)

input_shape = dataset.input_shape
output_size = dataset.output_size

steps_per_epoch = dataset.size / BATCH_SIZE

voc = Vocabulary()
voc.retrieve(output_path)

generator = Generator.data_generator(voc, gui_paths, img_paths, batch_size=BATCH_SIZE,
                                     generate_binary_sequences=True)

# val_dataset = Dataset()
# val_dataset.load(val_path, generate_binary_sequences=True,
#                  data_percentage=data_percentage)
# val_dataset.save_metadata(output_path)
# val_dataset.voc.save(output_path)

# val_gui_paths, val_img_paths = Dataset.load_paths_only(val_path)

# val_input_shape = val_dataset.input_shape
# val_output_size = val_dataset.output_size

# val_steps_per_epoch = val_dataset.size / BATCH_SIZE

# val_voc = Vocabulary()
# val_voc.retrieve(output_path)

# val_generator = Generator.data_generator(val_voc, val_gui_paths, val_img_paths,
#                                          batch_size=BATCH_SIZE,
#                                          generate_binary_sequences=True)

from keras import backend as K
K.clear_session()
dropout_rate = 0.25
opt = Adam(lr=1e-3)
dopt = Adam(lr=1e-4)

n_channels = 32
g_input = Input(shape=[4])
H = Dense(n_channels*32*32, init='glorot_normal')(g_input)
H = BatchNormalization()(H)
H = Activation('relu')(H)
H = Reshape( [n_channels, 32, 32] )(H)
H = UpSampling2D(size=(2, 2))(H)
H = Convolution2D(int(n_channels/2), 3, 3, border_mode='same', init='glorot_uniform')(H)
H = BatchNormalization()(H)
H = Activation('relu')(H)
H = UpSampling2D(size=(2, 2))(H)
H = Convolution2D(int(n_channels/4), 3, 3, border_mode='same', init='glorot_uniform')(H)
H = BatchNormalization()(H)
H = Activation('relu')(H)
H = Convolution2D(3, 1, 1, border_mode='same', init='glorot_uniform')(H)
g_V = Activation('sigmoid')(H)
generator_model = Model(g_input,g_V)
generator_model.compile(loss='binary_crossentropy', optimizer=opt)


d_input = Input(shape=(128,128,3))
H = Convolution2D(32, (3, 3), padding='same', activation='relu')(d_input)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Convolution2D(64, (3, 3), padding='same', activation='relu')(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Flatten()(H)
H = Dense(256)(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
d_V = Dense(2,activation='softmax')(H)
discriminator = Model(d_input,d_V)
discriminator.compile(loss='categorical_crossentropy', optimizer=dopt)

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


make_trainable(discriminator, False)
# Build stacked GAN model
gan_input = Input(shape=[4])
H = generator_model(gan_input)
gan_V = discriminator(H)
GAN = Model(gan_input, gan_V)
GAN.compile(loss='categorical_crossentropy', optimizer=opt)

#print(generated_images.shape)

losses = {"d":[], "g":[]}
plt_frq = 50
for e in tqdm(range(100)):
    print("Epoch nb {}".format(e))
    # Make generative images
    X_y, y_1 = next(generator)
    Xtrain, ytrain = X_y[0], X_y[1]
    resized = []
    for udx, img in enumerate(Xtrain):
        n_img = cv2.resize(img,(128, 128))
        resized.append(n_img)

    resized = np.array(resized)
    noise_gen = np.random.uniform(0,1,size=[4,4])
    generated_images = generator_model.predict(noise_gen)
    
    # Train discriminator on generated images
    X = np.concatenate((resized[:2,:,:,:], generated_images[:2,:,:,:]))
    n = 2
    y = np.zeros([2*n,2])
    y[:n,1] = 1
    y[n:,0] = 1

    make_trainable(discriminator,True)
    d_loss  = discriminator.train_on_batch(X,y)
    losses["d"].append(d_loss)
    
    # train Generator-Discriminator stack on input noise to non-generated output class
    noise_tr = np.random.uniform(0,1,size=[4,4])
    y2 = np.zeros([4,2])
    y2[:,1] = 1
    make_trainable(discriminator,False)
    g_loss = GAN.train_on_batch(noise_tr, y2)
    losses["g"].append(g_loss)

    # Updates plots
    if e%plt_frq==plt_frq-1:
        plt.imshow(generated_images[0])
        plt.savefig("generated_ex_epoch_{}.png".format(e), bbox_inches="tight")
        print("losses g {}".format(losses['g']))
        print("losses d {}".format(losses['d']))

GAN.save("Gan.h5")
print("FINISHED!!")