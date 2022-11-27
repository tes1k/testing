import os

windows = (True if (os.name == 'nt') else False)
if windows:
    OS = 'windows'
else:
    OS = 'linux'

MAIN_PATH = str(os.path.dirname(os.path.abspath(__file__)).split('source')[0])  # Path of the folder outside source

if windows:
    INPUTS_PATH = os.path.join(MAIN_PATH, "inputs")
    OUTPUT_FOLDER_PATH = os.path.join(MAIN_PATH, "outputs")

else:
    INPUTS_PATH = "./ml_vol/inputs/"
    OUTPUT_FOLDER_PATH = "./ml_vol/outputs"

TRAIN_FILE_SUBSTRING = "_train.csv"
TEST_FILE_SUBSTRING = "_test_key.csv"
SCHEMA_FILE_SUBSTRING = "_schema.json"

DATA_FOLDER = os.path.join(INPUTS_PATH, "data")
TRAIN_DATA_FOLDER = os.path.join(DATA_FOLDER, "training")
TEST_DATA_FOLDER = os.path.join(DATA_FOLDER, "testing")
SAVE_FOLDER = os.path.join(MAIN_PATH, "save_files")

DATA_SCHEMA_PATH = os.path.join(INPUTS_PATH, "data_config")
TESTING_OUTPUTS_PATH = os.path.join(OUTPUT_FOLDER_PATH, "testing_outputs")

MODEL_SAVE_PATH = os.path.join(SAVE_FOLDER, "my_model.h5")
TUNER_SAVE_PATH = os.path.join(SAVE_FOLDER, "tuner")
NORM_SCALAR_SAVE_PATH = os.path.join(SAVE_FOLDER, "min_max_scalar_data.npy")
BEST_HP_SAVE_PATH = os.path.join(SAVE_FOLDER, "best_hyperparameters.pckl")
BEST_HP_JSON_SAVE_PATH = os.path.join(SAVE_FOLDER, "best_hyperparameters.json")
HP_TUNE_RANGE_JSON_SAVE_PATH = os.path.join(SAVE_FOLDER, "hyparameter_tune_range.json")
TRAINING_USER_CHOICE_SAVE_PATH = os.path.join(SAVE_FOLDER, "user_choice.json")

# ---PERMANANT HYPARAMETERS---------------------------------------------
shuffle_buffer_size = 1000
test_split_percent = 5
max_epochs = 10000  # max number of epochs.
lr = 0.001  # Initial lr when training
# ----------------------------------------------------------------------

# ---HYPERPARAMETERS TUNE PARAMETERS------------------------------------
target_tune_steps = 100000
max_trails = 5
# ----------------------------------------------------------------------

# ---NAMES--------------------------------------------------------
LSTM_model_name = 'LSTM_model'
RNN_model_name = 'RNN_model'
GRU_model_name = 'GRU_model'
windows_name = 'windows'
linux_name = 'linux'
# ----------------------------------------------------------------------
