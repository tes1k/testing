from keras.layers import *
import tensorflow as tf
from keras.callbacks import *
import matplotlib.pyplot as plt
import keras_tuner
from keras.models import *

import source.utils as utils
import source.config as config
import source.load_data as data

shuffle_buffer_size = config.shuffle_buffer_size
test_split_percent = config.test_split_percent
max_epochs = config.max_epochs
lr = config.lr
enable_validation = True

tune_hp_ranges = {}


# Override inbuilt class to tune the preprocessor
class MyHyperModel(keras_tuner.HyperModel):
    def build(self, hp):
        model = def_tuning_model(hp)
        model.summary()
        return model

    def fit(self, hp, model, **kwargs):
        tf.keras.backend.clear_session()

        window_size_range = tune_hp_ranges['window_size_range']
        if len(window_size_range) > 1:
            window_size = hp.Int('window_size',
                                 min_value=window_size_range[0],
                                 max_value=window_size_range[-1],
                                 step=(window_size_range[1] - window_size_range[0]))
        else:
            window_size = window_size_range[0]

        batch_size_range = tune_hp_ranges['batch_size_range']
        if len(batch_size_range) > 1:
            batch_size = hp.Int('batch_size',
                                   min_value=batch_size_range[0],
                                   max_value=batch_size_range[-1],
                                   step=(batch_size_range[1] - batch_size_range[0])
                                   )
        else:
            batch_size = batch_size_range[0]

        train_dataset, val_dataset = data.load_preprocess_data(enable_validation=enable_validation,
                                                               window_size=window_size,
                                                               batch_size=batch_size,
                                                               shuffle_buffer_size=shuffle_buffer_size)

        print("\nfit()\nTuning ranges list:")
        for element in tune_hp_ranges:
            print(element, tune_hp_ranges[element])

        print(f"\nfit()-\n"
              f"--Window size={window_size}\n"
              f"--Batch size={batch_size}")

        return model.fit(
            x=train_dataset,
            epochs=tune_hp_ranges['tune_epochs'],  # Disable if using HyperBand
            # validation_data=val_dataset,  # Not feeding val data because we don't use it to set hyperparameters
            **kwargs,
        )


def def_tuning_model(hp):
    # Hyperparameter choosers below simply return an variable like int, bool, etc
    units_range = tune_hp_ranges['units_range']
    if len(units_range) > 1:
        units = hp.Int('units',
                               min_value=units_range[0],
                                 max_value=units_range[-1],
                                 step=(units_range[1] - units_range[0]))
    else:
        units = units_range[0]

    num_layers_range = tune_hp_ranges['num_layers_range']
    if len(num_layers_range) > 1:
        num_layers = hp.Int('num_layers',
                               min_value=num_layers_range[0],
                                 max_value=num_layers_range[-1],
                                 step=(num_layers_range[1] - num_layers_range[0]))
    else:
        num_layers = num_layers_range[0]

    print(f"def_tuning_model()\n"
          f"units= {units}\n"
          # f"Dense_units= {Dense_units}"
          )

    # lr is not tuned because we use ReduceLROnPlateau()
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    metrics = ["mae", "mse"]

    model = def_model(units=units, num_layers=num_layers)

    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=metrics,
                  run_eagerly=True)

    return model


# Call this function to automatically tune the hyperparameters
def tune_hyperparameters(continue_training=False):
    global tune_hp_ranges

    tf.random.set_seed(51)
    np.random.seed(51)
    tf.keras.backend.clear_session()

    utils.set_hp_tuning_range(train_data_list=data.load_train_from_file())

    tune_hp_ranges = utils.load_dict_from_json(config.HP_TUNE_RANGE_JSON_SAVE_PATH)
    print(f"tune_hyperparameters()\ntune_hp_ranges:\n{tune_hp_ranges}")

    tuner = keras_tuner.BayesianOptimization(
        MyHyperModel(),
        objective='loss',
        executions_per_trial=1,
        project_name=config.TUNER_SAVE_PATH,
        overwrite=not continue_training,
        max_trials=config.max_trails,
    )

    tuner.search_space_summary()

    tuner_history = tuner.search()

    best_hyperparameters = tuner.get_best_hyperparameters()[0].values
    best_hyperparameters["tuning_done_for"] = utils.load_dict_from_json(config.TRAINING_USER_CHOICE_SAVE_PATH)["id_name"]
    print("Best hyperparameters:\n", best_hyperparameters)

    # keras docs recommends we retrain using the best parameters. Use the saved hyperparameters to retrain normally using train_model()
    utils.save_dict_as_json(config.BEST_HP_JSON_SAVE_PATH, best_hyperparameters)


# Model definition is changed at every trail of the tuner. Best hyperparameters can later be called in a similar way
def def_LSTM_model(LSTM_units, num_LSTM_layers, dense_units=1):
    X_Input = Input(shape=[None])

    X = Lambda(lambda x: tf.expand_dims(x, axis=-1))(X_Input)

    for i in range(num_LSTM_layers):
        X = LSTM(units=LSTM_units, return_sequences=True)(X)

    X = LSTM(units=LSTM_units)(X)

    X = Dense(1)(X)

    model = Model(inputs=X_Input, outputs=X, name=config.LSTM_model_name)

    return model


def def_RNN_model(RNN_units, num_RNN_layers, dense_units=1):
    X_Input = Input(shape=[None])

    X = Lambda(lambda x: tf.expand_dims(x, axis=-1))(X_Input)

    for i in range(num_RNN_layers):
        X = SimpleRNN(RNN_units, return_sequences=True)(X)

    X = SimpleRNN(RNN_units)(X)

    X = Dense(1)(X)

    model = Model(inputs=X_Input, outputs=X, name=config.RNN_model_name)

    return model


def def_GRU_model(GRU_units, num_GRU_layers, dense_units=1):
    X_Input = Input(shape=[None])

    X = Lambda(lambda x: tf.expand_dims(x, axis=-1))(X_Input)

    for i in range(num_GRU_layers):
        X = GRU(GRU_units, return_sequences=True)(X)

    X = GRU(GRU_units)(X)

    X = Dense(1)(X)

    model = Model(inputs=X_Input, outputs=X, name=config.GRU_model_name)

    return model


def def_model(units, num_layers):
    model_choice = utils.load_dict_from_json(config.TRAINING_USER_CHOICE_SAVE_PATH)["model_choice"]
    if model_choice == config.LSTM_model_name:
        return def_LSTM_model(LSTM_units=units, num_LSTM_layers=num_layers)
    elif model_choice == config.RNN_model_name:
        return def_RNN_model(RNN_units=units, num_RNN_layers=num_layers)
    elif model_choice == config.GRU_model_name:
        return def_GRU_model(GRU_units=units, num_GRU_layers=num_layers)


# Once the best hyperparameters are found using tune_hyperparameters(), call this function to train the model
def train_model():
    global tune_hp_ranges

    utils.read_schema()
    utils.set_hp_tuning_range(train_data_list=data.load_train_from_file())

    best_hyperparameters = utils.load_dict_from_json(config.BEST_HP_JSON_SAVE_PATH)
    print(f"Best hyperparameters, {best_hyperparameters}")

    tune_hp_ranges = utils.load_dict_from_json(config.HP_TUNE_RANGE_JSON_SAVE_PATH)
    print("tune_hp_ranges", tune_hp_ranges)

    # Get the best hyperparameters if it is not disabled. Tuning is disabled if range a single array
    window_size = tune_hp_ranges["window_size_range"][0] if (len(tune_hp_ranges["window_size_range"]) == 1) else best_hyperparameters['window_size']
    batch_size = tune_hp_ranges["batch_size_range"][0] if (len(tune_hp_ranges["batch_size_range"]) == 1) else best_hyperparameters['batch_size']
    # dense_units = tune_hp_ranges["dense_range"][0] if (len(tune_hp_ranges["dense_range"]) == 1) else best_hyperparameters['dense_units']
    units = tune_hp_ranges["units_range"][0] if (len(tune_hp_ranges["units_range"]) == 1) else best_hyperparameters['units']
    num_layers = tune_hp_ranges["num_layers_range"][0] if (len(tune_hp_ranges["num_layers_range"]) == 1) else best_hyperparameters['num_layers']

    train_dataset, val_dataset = data.load_preprocess_data(enable_validation=enable_validation,
                                                           window_size=window_size,
                                                           batch_size=batch_size,
                                                           shuffle_buffer_size=shuffle_buffer_size)

    model = def_model(units=units, num_layers=num_layers)
    model.summary()

    Reducing_LR = ReduceLROnPlateau(monitor='loss',
                                    factor=0.2,
                                    min_delta=1e-8,
                                    patience=1 * (tune_hp_ranges['tune_epochs']),
                                    verbose=1)

    Early_Stopping = EarlyStopping(monitor='loss',
                                   patience=4 * (tune_hp_ranges['tune_epochs']),
                                   restore_best_weights=True,
                                   min_delta=1e-8,
                                   verbose=1)

    optimizer = tf.keras.optimizers.Adam(lr=lr)
    metrics = ["mae", "mse"]
    callbacks = [Reducing_LR, Early_Stopping]

    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=metrics)

    history = model.fit(x=train_dataset, epochs=max_epochs, callbacks=callbacks, validation_data=val_dataset, verbose=1)
    model.save(config.MODEL_SAVE_PATH)

    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Mean absolute error')
    plt.ylabel('')
    plt.xlabel('epoch')
    plt.legend(['Train set', 'Test set'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('')
    plt.legend(['loss', 'val_loss'], loc='upper left')
    plt.show()


if len(sys.argv) > 1:
    ch = sys.argv[1]
    if ch == 'train':
        train_model()
    elif ch == 'tune':
        tune_hyperparameters(continue_training=False)
    else:
        print("Please enter the right second argument. It's either 'train' or 'test'")