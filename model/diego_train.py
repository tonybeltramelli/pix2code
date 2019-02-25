#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

import sys
import json

from classes.dataset.Generator import *
from classes.model.pix2code import *
from classes.model.shallow_pix2code import *
from classes.model.attention_pix2code import attention_pix2code
from evaluation import calculate_set_levenshtein_distance


def run(input_path, output_path, which_model, epochs, data_percentage, val_path):
    np.random.seed(1234)

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

    val_dataset = Dataset()
    val_dataset.load(val_path, generate_binary_sequences=True,
                     data_percentage=data_percentage)
    val_dataset.save_metadata(output_path)
    val_dataset.voc.save(output_path)

    val_gui_paths, val_img_paths = Dataset.load_paths_only(val_path)

    val_input_shape = val_dataset.input_shape
    val_output_size = val_dataset.output_size

    val_steps_per_epoch = val_dataset.size / BATCH_SIZE

    val_voc = Vocabulary()
    val_voc.retrieve(output_path)

    val_generator = Generator.data_generator(val_voc, val_gui_paths, val_img_paths,
                                             batch_size=BATCH_SIZE,
                                             generate_binary_sequences=True)

    if which_model == 'shallow':
        model = shallow_pix2code(input_shape, output_size, output_path)
    elif which_model == 'attention':
        model = attention_pix2code(input_shape, output_size, output_path)
    else:
        model = pix2code(input_shape, output_size, output_path)

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
    for epoch in range(epochs):
        callbacks = model.model.fit_generator(generator, callbacks=[checkpoint],
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

        val_lev_distance.append(calculate_set_levenshtein_distance(test_path,
                                                                   output_path,
                                                                   model.model))

        with open(loss_file_name, 'w') as file_:
            json.dump({
                "train_loss": train_loss_per_epoch,
                'train_lev_distance': train_lev_distance, "val_loss": val_loss_per_epoch,
                'val_lev_distance': val_lev_distance, 'train_accuracy': train_acc_per_epoch,
                'val_accuracy': val_acc_per_epoch
            }, file_)

    model.model.save('saved_models/{}'.format(snapshot_name))


if __name__ == "__main__":
    argv = sys.argv[1:]

    if len(argv) < 2:
        print("Error: not enough argument supplied:")
        print("train.py <input path> <output path> "
              "<shallow or pix2code or attention > <epochs> <data_percentage>")
        exit(0)
    else:
        input_path = argv[0]
        output_path = argv[1]
        which_model = argv[2]
        if which_model not in ['shallow', 'pix2code', 'attention']:
            raise ValueError("model choice should be either 'shallow' or 'pix2code")
        epochs = int(argv[3]) if len(argv) > 2 else 20
        data_percentage = float(argv[4]) if len(argv) > 3 else 1.0
        val_path = argv[5]

    run(input_path, output_path, which_model, epochs, data_percentage, val_path)
