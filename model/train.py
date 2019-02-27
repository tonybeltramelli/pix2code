#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import
__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

import sys
import json

from keras.callbacks import ModelCheckpoint

from classes.dataset.Generator import *
from classes.model.pix2code import *


def run(input_path, output_path, val_path, is_memory_intensive=False,
        pretrained_model=None):
    np.random.seed(1234)

    dataset = Dataset()
    dataset.load(input_path, generate_binary_sequences=True)
    dataset.save_metadata(output_path)
    dataset.voc.save(output_path)

    val_dataset = Dataset()
    val_dataset.load(val_path, generate_binary_sequences=True)
    val_dataset.save_metadata(output_path)
    val_dataset.voc.save(output_path)

    if not is_memory_intensive:
        dataset.convert_arrays()

        input_shape = dataset.input_shape
        output_size = dataset.output_size
        print(len(dataset.input_images), len(dataset.partial_sequences), len(dataset.next_words))
        print(dataset.input_images.shape, dataset.partial_sequences.shape, dataset.next_words.shape)
    else:
        gui_paths, img_paths = Dataset.load_paths_only(input_path)
        input_shape = dataset.input_shape
        output_size = dataset.output_size
        steps_per_epoch = dataset.size / BATCH_SIZE

        voc = Vocabulary()
        voc.retrieve(output_path)

        generator = Generator.data_generator(voc, gui_paths, img_paths, batch_size=BATCH_SIZE, generate_binary_sequences=True)
        val_gui_paths, val_img_paths = Dataset.load_paths_only(val_path)


        val_steps_per_epoch = val_dataset.size / BATCH_SIZE

        val_voc = Vocabulary()
        val_voc.retrieve(output_path)

        val_generator = Generator.data_generator(val_voc, val_gui_paths, val_img_paths,
                                                 batch_size=BATCH_SIZE,
                                                 generate_binary_sequences=True)


    model = pix2code(input_shape, output_size, output_path)

    if pretrained_model is not None:
        model.model.load_weights(pretrained_model)
    train_loss_per_epoch = list()
    val_loss_per_epoch = list()
    train_lev_distance = list()
    val_lev_distance = list()
    train_acc_per_epoch = list()
    val_acc_per_epoch = list()

    experiment_name = '{}_model'.format(model.name)
    loss_file_name = 'loss_outputs/loss_history_{}_model.json'.format(experiment_name)
    file_prepath = "model-{}".format(model.name)

    filepath = file_prepath + "epoch-{epoch:02d}-val-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss',
                                 verbose=1, save_best_only=True)
    if not is_memory_intensive:
        model.fit(dataset.input_images, dataset.partial_sequences, dataset.next_words)
    else:
        print("must print this! ASDFSFASDFA")
        for _ in range(25):
            model.fit_generator(generator, callbacks=[checkpoint],
                                steps_per_epoch=steps_per_epoch, epochs=1,
                                validation_data=val_generator,
                                validation_steps=val_steps_per_epoch)
            loss = callbacks.history['loss'][0]
            acc = callbacks.history['acc'][0]
            val_loss = callbacks.history['val_loss'][0]
            val_acc = callbacks.history['val_acc'][0]

            train_loss_per_epoch.append(loss)
            val_loss_per_epoch.append(val_loss)

            train_acc_per_epoch.append(acc)
            val_acc_per_epoch.append(val_acc)
            snapshot_name = '{}_epoch_{}.h5'.format(experiment_name, epoch)

            train_lev_distance.append(calculate_set_levenshtein_distance(input_path,
                                                                         output_path,
                                                                         model.model))

            val_lev_distance.append(calculate_set_levenshtein_distance(val_path,
                                                                       output_path,
                                                                       model.model))

            with open(loss_file_name, 'w') as file_:
                json.dump({
                    "train_loss": train_loss_per_epoch,
                    'train_lev_distance': train_lev_distance,
                    "val_loss": val_loss_per_epoch,
                    'val_lev_distance': val_lev_distance,
                    'train_accuracy': train_acc_per_epoch,
                    'val_accuracy': val_acc_per_epoch
                }, file_)

if __name__ == "__main__":
    argv = sys.argv[1:]

    if len(argv) < 2:
        print("Error: not enough argument supplied:")
        print("train.py <input path> <output path> <is memory intensive (default: 0)> <pretrained weights (optional)>")
        exit(0)
    else:
        input_path = argv[0]
        output_path = argv[1]
        val_path = argv[2]
        use_generator = False if len(argv) < 4 else True if int(argv[3]) == 1 else False
        pretrained_weigths = None if len(argv) < 5 else argv[4]

    run(input_path, output_path, val_path, is_memory_intensive=use_generator, pretrained_model=pretrained_weigths)
