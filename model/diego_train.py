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


def run(input_path, output_path, which_model, epochs, data_percentage, test_path):
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

    #steps_per_epoch = steps_per_epoch if steps_per_epoch > 0 else 1

    voc = Vocabulary()
    voc.retrieve(output_path)

    generator = Generator.data_generator(voc, gui_paths, img_paths, batch_size=BATCH_SIZE,
                                         generate_binary_sequences=True)
    if which_model == 'shallow':
        model = shallow_pix2code(input_shape, output_size, output_path)
    elif which_model == 'attention':
        model = attention_pix2code(input_shape, output_size, output_path)
    else:
        model = pix2code(input_shape, output_size, output_path)

    loss_per_epoch = list()
    lev_distance = list()
    experiment_name = '{}_{}_model'.format(model.name, data_percentage)
    loss_file_name = 'loss_outputs/loss_history_{}_model.json'.format(experiment_name)
    for epoch in range(epochs):
        callbacks = model.model.fit_generator(generator,
                                              steps_per_epoch=steps_per_epoch,
                                              epochs=1)
        loss = callbacks.history['loss'][0]
        loss_per_epoch.append(loss)
        snapshot_name = '{}_epoch_{}.h5'.format(experiment_name, epoch)
        model.model.save('saved_models/{}'.format(snapshot_name))

        lev_distance.append(calculate_set_levenshtein_distance(test_path,
                                                               output_path,
                                                               model.model))

        with open(loss_file_name, 'w') as file:
            json.dump({"loss": loss_per_epoch, 'lev_distance': lev_distance}, file)


if __name__ == "__main__":
    argv = sys.argv[1:]

    if len(argv) < 2:
        print("Error: not enough argument supplied:")
        print("train.py <input path> <output path> "
              "<shallow or pix2code> <epochs> <data_percentage>")
        exit(0)
    else:
        input_path = argv[0]
        output_path = argv[1]
        which_model = argv[2]
        if which_model not in ['shallow', 'pix2code', 'attention']:
            raise ValueError("model choice should be either 'shallow' or 'pix2code")
        epochs = int(argv[3]) if len(argv) > 2 else 10
        data_percentage = float(argv[4]) if len(argv) > 3 else 1.0
        test_path = argv[5]

    run(input_path, output_path, which_model, epochs, data_percentage, test_path)
