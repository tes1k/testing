import os.path
import traceback
import numpy as np
import pickle
import json
import math
import csv

import source.config as config


def save_dict_as_json(file_name, dict, over_write=False):
    if os.path.exists(file_name) and over_write:
        with open(file_name) as f:
            existing_dict = json.load(f)
            # print(existing_dict)

        existing_dict.update(dict)

        # print(f"Saving json file name {file_name}\n Data:\n{existing_dict}")
        with open(file_name, 'w') as f:
            json.dump(existing_dict, f)
    else:
        # print(f"Saving json file name {file_name}\n Data:\n{dict}")
        with open(file_name, 'w') as f:
            json.dump(dict, f)


def load_dict_from_json(file_name):
    with open(file_name) as f:
        d = json.load(f)
        # print(d)

    return d

# save pickle object
def save_object(file_name, object):
    f = open(file_name, 'wb')
    pickle.dump(object, f)
    f.close()

# load pickle object
def load_object(file_name):
    f = open(file_name, 'rb')
    object = pickle.load(f)
    return object


def find_filename_match(known_filename, directory):
    files_list = os.listdir(directory)
    for file_name in files_list:
        if known_filename in file_name:
            return os.path.join(directory, file_name)


# Goal of this function is to return a number between 1 and 0. Default is division with max, but you can use minmax_scaler as well
def normalise(list, save_min_max=False, norm_range=(0, 1), use_minmax=False):

    X = np.array(list).squeeze()

    min_norm_range, max_norm_range = norm_range

    if save_min_max:
        X_min = np.min(X)
        X_max = np.max(X)
    else:
        try:
            (X_min, X_max) = load_object(config.NORM_SCALAR_SAVE_PATH)
        except:
            traceback.print_exc()
            raise Exception("Make sure NORM_SCALAR was saved correctly from train data during training")

    if use_minmax:
        X_std = (X - X_min) / (X_max - X_min)
        X_norm = X_std * (max_norm_range - min_norm_range) + min_norm_range
    else:
        # Divide by max value
        X_norm = X / X_max

    print('Min: %f, Max: %f' % (X_min, X_max), ("minmax=", use_minmax))

    if save_min_max:
        save_object(config.NORM_SCALAR_SAVE_PATH, (X_min, X_max))

    return X_norm


def inverse_normalise(list, use_minmax=False):

    X = np.array(list).squeeze()
    (X_min, X_max) = load_object(config.NORM_SCALAR_SAVE_PATH)

    if use_minmax:
        og_val = (X * (X_max - X_min)) + X_min
    else:
        og_val = X * X_max

    return og_val


# Sets the range of hyperparameters that are tested based on the length of the data
def set_hp_tuning_range(train_data_list):
    def find_nearest_pow_2(num):
        power = 1
        while (power < num):
            power *= 2
        return power

    model_choice = load_dict_from_json(config.TRAINING_USER_CHOICE_SAVE_PATH)["model_choice"]
    train_data_len = len(train_data_list)
    small_list_data_len = 5000  # Lists with length smaller than this can be deemed as too small

    min_batch_size = int(find_nearest_pow_2(math.sqrt(train_data_len)) / 2)  # Set batch size as closest to sqrt of train_data_len
    batch_size_range = [min_batch_size]
    window_size_range = []
    if (train_data_len > small_list_data_len):
        for window in range(100, min(500, round(train_data_len / 3)), 200):
            window_size_range.append(window)
    else:
        window_size_range = list(range(30, min(round(train_data_len / 2), 90), 30))


    units_range = [64, 128]
    num_layers_range = list(range(1, 6))  # 2 to 5 layers

    if train_data_len < small_list_data_len / 2:
        epochs = 25
    elif train_data_len < small_list_data_len:
        epochs = 12
    else:
        epochs = max(round(config.target_tune_steps / train_data_len), 2)  # Run at least twice

    if model_choice == config.RNN_model_name:
        epochs = round(epochs / 4)


    hp_tune_range_dict = {'batch_size_range': batch_size_range,
                            'window_size_range': window_size_range,
                            'units_range': units_range,
                            'num_layers_range': num_layers_range,
                            # 'dense_range': dense_range,
                            'tune_epochs': epochs,  # This is the number of times a single trail of tuning will run. This value will also be used for lr auto reduce
                          }

    def set_def_hp():
        print(f"------------SETTING DEFAULT HP FOR TRAINING {load_dict_from_json(config.TRAINING_USER_CHOICE_SAVE_PATH)['id_name']}")
        best_hp = {"units": units_range[1],
                   "num_layers": 2,
                   "window_size": window_size_range[0],
                   "tuning_done_for": load_dict_from_json(config.TRAINING_USER_CHOICE_SAVE_PATH)["id_name"]}
        save_dict_as_json(config.BEST_HP_JSON_SAVE_PATH, best_hp)

    # Set default hp if tuning was skipped
    if os.path.exists(config.BEST_HP_JSON_SAVE_PATH):
        best_hp = load_dict_from_json(config.BEST_HP_JSON_SAVE_PATH)
        if not ("tuning_done_for" in best_hp.keys()):
            print("No tuning_done_for detected with the best_hyperparameters")
            set_def_hp()

        elif best_hp["tuning_done_for"] != load_dict_from_json(config.TRAINING_USER_CHOICE_SAVE_PATH)["id_name"]:
            print(f"New id_name {load_dict_from_json(config.TRAINING_USER_CHOICE_SAVE_PATH)['id_name']} found instead of old, {best_hp['tuning_done_for']}")
            set_def_hp()
    else:
        print("No best_hyperparameters file found")
        set_def_hp()


    save_dict_as_json(config.HP_TUNE_RANGE_JSON_SAVE_PATH, hp_tune_range_dict)


# Call this to read and store column number from the schema. This also sets a default id_name if none is detected
def read_schema(train_data_file_path=find_filename_match(known_filename=config.TRAIN_FILE_SUBSTRING, directory=config.TRAIN_DATA_FOLDER)):
    train_csv_data = list(csv.reader(open(train_data_file_path, 'r')))

    column_name_list = []
    for i, column in enumerate(train_csv_data[0]):
        print(str(i + 1) + ": ", column, end=" | ")
        column_name_list.append(column)
    print()

    schema_filename = find_filename_match(directory=config.INPUTS_PATH, known_filename=config.SCHEMA_FILE_SUBSTRING)
    schema_json = load_dict_from_json(schema_filename)
    data_column_number = column_name_list.index(schema_json["inputDatasets"]["forecastingBaseHistory"]["targetField"])
    print("Chosen data column: ", column_name_list[data_column_number])

    try:
        user_choices = load_dict_from_json(config.TRAINING_USER_CHOICE_SAVE_PATH)
    except:
        user_choices = {}

    id_name_list = extract_id_names()

    user_choices["data_column_number"] = data_column_number
    if "id_name" in user_choices.keys():
        if not (user_choices["id_name"] in id_name_list):
            print(f"--------SETTING DEFAULT id_name {id_name_list[0]} in user_choice.json AS NO VALID CHOICE IS DETECTED----------")
            user_choices["id_name"] = id_name_list[0]  # Set a default id_name if there was no id name chosen by the user, or if the one chosen is not from the list

    save_dict_as_json(config.TRAINING_USER_CHOICE_SAVE_PATH, user_choices)



# Ask user which name should be chosen to train, if there are many.
# Example, if there are multiple stock data in one file, a list would be displayed asking the user to enter which stock they wish to use
def extract_id_names(train_data_file_path=find_filename_match(known_filename=config.TRAIN_FILE_SUBSTRING, directory=config.TRAIN_DATA_FOLDER)):
    train_csv_data = list(csv.reader(open(train_data_file_path, 'r')))

    id_name_list = []
    id_name_list_cnt = [0, 0]
    last_name = ""
    current_index_cnt = 0
    for i, data in enumerate(train_csv_data):
        if i == 0:
            continue

        if not data[0] in id_name_list:
            id_name_list.append(data[0])
            id_name_list_cnt[-1] = current_index_cnt
            id_name_list_cnt.append(0)
            current_index_cnt = 1
        else:
            current_index_cnt += 1

    return id_name_list


# Store the column of the csv file that is used for forcasting, and the symbol of the item you wish to forecast, if there are multiple
def read_csv_store_user_choice(train_data_file_path=find_filename_match(known_filename=config.TRAIN_FILE_SUBSTRING, directory=config.TRAIN_DATA_FOLDER)):
    global data_column_number

    read_schema()

    id_name_list = extract_id_names()

    print("\n")
    print("Index\tName")
    for i, name in enumerate(id_name_list):
        print(str(i + 1) + ":\t", name)

    chosen_list_num = 0
    if len(id_name_list) > 1:
        chosen_list_num = int(input("\nEnter the number which is followed by the list you which you wish to forecast: ")) - 1
        print("Chosen list name:", id_name_list[chosen_list_num])

    # Ask user which model type they wish to use
    model_choice = int(input("\n\n\n1) LSTM\n"
                             "2) RNN\n"
                             "3) GRU\n"
                             "Please choose the model type you wish use: "))
    if model_choice == 1:
        model_choice = config.LSTM_model_name
    elif model_choice == 2:
        model_choice = config.RNN_model_name
    elif model_choice == 3:
        model_choice = config.GRU_model_name

    try:
        user_choices = load_dict_from_json(config.TRAINING_USER_CHOICE_SAVE_PATH)
    except:
        user_choices = {}

    user_choices["data_column_number"] = data_column_number
    user_choices["id_name"] =  id_name_list[chosen_list_num]
    user_choices["model_choice"] = model_choice
    save_dict_as_json(config.TRAINING_USER_CHOICE_SAVE_PATH, user_choices)

    print("User choices:\n")
    print(user_choices)