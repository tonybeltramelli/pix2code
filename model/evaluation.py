from string import ascii_lowercase

from keras.models import load_model
from Levenshtein import distance

from classes.dataset.Generator import *
from classes.diego_sampler import *


def load_dataset_and_vocabulary(set_path, output_path):
    dataset = Dataset()
    dataset.load(set_path, generate_binary_sequences=True)
    dataset.save_metadata(output_path)
    dataset.voc.save(output_path)

    gui_paths, img_paths = Dataset.load_paths_only(set_path)
    voc = Vocabulary()
    voc.retrieve(output_path)
    return gui_paths, img_paths, voc


def generate_sequence(voc, gui_file):
    token_sequence = [START_TOKEN]
    for line in gui_file:
        line = line.replace(",", " ,").replace("\n", " \n")
        tokens = line.split(" ")
        for token in tokens:
            voc.append(token)
            token_sequence.append(token)
    token_sequence.append(END_TOKEN)
    return token_sequence, voc


def read_image_input(image_path):
    if image_path.find(".png") != -1:
        return Utils.get_preprocessed_img(image_path, IMAGE_SIZE)
    else:
        return np.load(image_path)["features"]


def token_char_mapping(vocab):
    mapping = {}
    sorted_ids = sorted(vocab.vocabulary.values())

    for id_ in sorted_ids:
        mapping[ascii_lowercase[id_]] = vocab.token_lookup[id_]

    inverse_mapping = {v: k for k, v in mapping.items()}
    return mapping, inverse_mapping


def calculate_set_levenshtein_distance(test_path, output_path, model, verbose=False):
    gui_paths, img_paths, voc = load_dataset_and_vocabulary(test_path, output_path)
    _, token_char_map = token_char_mapping(voc)

    sampler = Sampler(voc_path='../bin', output_size=19, context_length=CONTEXT_LENGTH)

    set_average_distance = 0
    for i in range(0, len(gui_paths))[:3]:
        image_path = img_paths[i]
        img = read_image_input(image_path)

        gui = open(gui_paths[i], 'r')

        token_sequence, voc = generate_sequence(voc, gui)

        result, _ = sampler.predict_greedy(model, np.array([img]),
                                           prediction_as_list=True)
        string_result = " ".join(result)
        string_token_sequence = " ".join(token_sequence)

        char_token_sequence = "".join(token_char_map[x]
                                      for x in string_token_sequence.split())
        char_result_sequence = "".join(token_char_map[x]
                                       for x in string_result.split())

        if i % 200 == 0:
            print("here are some mappings")
            print("target sequence ", char_token_sequence)
            print("predicted sequence ", char_result_sequence)

        prediction_distance = distance(char_result_sequence, char_token_sequence)
        if verbose:
            print("Levenshtein distance {}".format(prediction_distance))

        if i == 0:
            set_average_distance = prediction_distance
        else:
            set_total_distance = set_average_distance * i
            set_average_distance = (set_total_distance + prediction_distance) / (i + 1)

        return set_average_distance
