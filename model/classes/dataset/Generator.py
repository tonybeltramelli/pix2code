from __future__ import print_function
__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import numpy as np

from classes.dataset.Dataset import *
from classes.Vocabulary import *
from classes.model.Config import *


class Generator:
    @staticmethod
    def data_generator(voc, gui_paths, img_paths, batch_size, generate_binary_sequences=False, verbose=False, loop_only_one=False):
        assert len(gui_paths) == len(img_paths)
        voc.create_binary_representation()

        while 1:
            batch_input_images = []
            batch_partial_sequences = []
            batch_next_words = []
            sample_in_batch_counter = 0

            for i in range(0, len(gui_paths)):
                if img_paths[i].find(".png") != -1:
                    img = Utils.get_preprocessed_img(img_paths[i], IMAGE_SIZE)
                else:
                    img = np.load(img_paths[i])["features"]
                gui = open(gui_paths[i], 'r')

                token_sequence = [START_TOKEN]
                for line in gui:
                    line = line.replace(",", " ,").replace("\n", " \n")
                    tokens = line.split(" ")
                    for token in tokens:
                        voc.append(token)
                        token_sequence.append(token)
                token_sequence.append(END_TOKEN)

                suffix = [PLACEHOLDER] * CONTEXT_LENGTH

                a = np.concatenate([suffix, token_sequence])
                for j in range(0, len(a) - CONTEXT_LENGTH):
                    context = a[j:j + CONTEXT_LENGTH]
                    label = a[j + CONTEXT_LENGTH]

                    batch_input_images.append(img)
                    batch_partial_sequences.append(context)
                    batch_next_words.append(label)
                    sample_in_batch_counter += 1

                    if sample_in_batch_counter == batch_size or (loop_only_one and i == len(gui_paths) - 1):
                        if verbose:
                            print("Generating sparse vectors...")
                        batch_next_words = Dataset.sparsify_labels(batch_next_words, voc)
                        if generate_binary_sequences:
                            batch_partial_sequences = Dataset.binarize(batch_partial_sequences, voc)
                        else:
                            batch_partial_sequences = Dataset.indexify(batch_partial_sequences, voc)

                        if verbose:
                            print("Convert arrays...")
                        batch_input_images = np.array(batch_input_images)
                        batch_partial_sequences = np.array(batch_partial_sequences)
                        batch_next_words = np.array(batch_next_words)

                        if verbose:
                            print("Yield batch")
                        yield ([batch_input_images, batch_partial_sequences], batch_next_words)

                        batch_input_images = []
                        batch_partial_sequences = []
                        batch_next_words = []
                        sample_in_batch_counter = 0
