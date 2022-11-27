import os
import source.config as config

def create_folder(new_path):
    if not os.path.exists(new_path):
        os.makedirs(new_path)

create_folder(config.INPUTS_PATH)
create_folder(config.DATA_FOLDER)
create_folder(config.TRAIN_DATA_FOLDER)
create_folder(config.TEST_DATA_FOLDER)
create_folder(config.OUTPUT_FOLDER_PATH)
create_folder(config.SAVE_FOLDER)