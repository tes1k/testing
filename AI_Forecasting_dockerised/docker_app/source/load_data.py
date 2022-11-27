import csv
import os
import tensorflow as tf
import math

import source.config as config
import source.utils as utils


data_column_number = None
test_split_percent = config.test_split_percent


def load_train_from_file(train_data_file_path=utils.find_filename_match(known_filename=config.TRAIN_FILE_SUBSTRING, directory=config.TRAIN_DATA_FOLDER)):
    global data_column_number

    data_column_number = utils.load_dict_from_json(config.TRAINING_USER_CHOICE_SAVE_PATH)["data_column_number"]
    chosen_id_name = utils.load_dict_from_json(config.TRAINING_USER_CHOICE_SAVE_PATH)["id_name"]

    train_csv_data = list(csv.reader(open(train_data_file_path, 'r')))

    train_series = []
    for i, data in enumerate(train_csv_data):
        if i == 0:
            continue

        if data[0] == chosen_id_name:
            # print(i, data)
            try:
                train_series.append(float(data[data_column_number]))
            except:
                pass

    print("len of loaded csv file ", len(train_series))
    return train_series


def load_csv_and_split():
    train_series = load_train_from_file()

    num_test = round((test_split_percent / 100) * len(train_series))
    print(f"num test {num_test}")

    test_series = train_series[-num_test:]
    train_series = train_series[:-num_test]

    return train_series, test_series


def load_test_from_file(test_data_file_path=utils.find_filename_match(known_filename=config.TEST_FILE_SUBSTRING, directory=config.TEST_DATA_FOLDER)):
    data_column_number = utils.load_dict_from_json(config.TRAINING_USER_CHOICE_SAVE_PATH)["data_column_number"]
    chosen_id_name = utils.load_dict_from_json(config.TRAINING_USER_CHOICE_SAVE_PATH)["id_name"]

    train_csv_data = list(csv.reader(open(test_data_file_path, 'r')))

    test_series = []
    for i, data in enumerate(train_csv_data):
        if i == 0:
            continue
        # print(i, data)

        if data[0] == chosen_id_name:
            try:
                test_series.append(float(data[data_column_number]))
            except:
                pass

    return test_series


def load_preprocess_data(enable_validation, window_size, batch_size, shuffle_buffer_size=config.shuffle_buffer_size):
    def is_test(x, _):
        test_modulus_number = round(100 / test_split_percent)  # returns every xth number a test. If 5% every 20th number
        return x % test_modulus_number == 0

    def is_train(x, y):
        return not is_test(x, y)

    def windowed_dataset(series, window_size, batch_size, shuffle_buffer, enable_validation):
        dataset = tf.data.Dataset.from_tensor_slices(series)
        dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
        dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
        dataset = dataset.batch(batch_size).prefetch(1)

        test_dataset = None
        if enable_validation:
            recover = lambda x, y: y
            # Split the dataset for training.
            test_dataset = dataset.enumerate() .filter(is_test).map(recover)
            # Split the dataset for testing/validation.
            train_dataset = dataset.enumerate().filter(is_train).map(recover)
            return train_dataset, test_dataset
        else:
            return dataset, test_dataset

    x_train = load_train_from_file()
    x_train = utils.normalise(x_train, save_min_max=True)

    train_dataset, test_dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size, enable_validation=enable_validation)

    print(f"load_preprocess_data()\nTrain dataset len= {len(list(train_dataset)) * batch_size}")
    print(f"Test dataset len= {len(list(test_dataset)) * batch_size}")

    return train_dataset, test_dataset