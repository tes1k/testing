import os.path
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
import csv

import source.load_data as data
import source.config as config
import source.utils as utils
import source.model as model_py


def forecast(test_keys_present=True, use_test_keys_for_forcast=False, num_days_arg=None):
    best_hyperparameters = utils.load_dict_from_json(config.BEST_HP_JSON_SAVE_PATH)
    print(f"Best hyperparameters, {best_hyperparameters}")

    tune_hp_ranges = utils.load_dict_from_json(config.HP_TUNE_RANGE_JSON_SAVE_PATH)
    print("tune_hp_ranges", tune_hp_ranges)

    # Get the best hyperparameters if it is not disabled. Tuning is disabled if range a single array
    window_size = tune_hp_ranges["window_size_range"][0] if (len(tune_hp_ranges["window_size_range"]) == 1) else best_hyperparameters['window_size']
    units = tune_hp_ranges["units_range"][0] if (len(tune_hp_ranges["units_range"]) == 1) else best_hyperparameters['units']
    num_layers = tune_hp_ranges["num_layers_range"][0] if (len(tune_hp_ranges["num_layers_range"]) == 1) else best_hyperparameters['num_layers']

    train_series = data.load_train_from_file()  # Known values
    if test_keys_present:
        test_series = data.load_test_from_file()  # Keys of the future values that will be forecasted
        test_series_norm = utils.normalise(test_series)
        # print("test_norm list: ", test_series_norm)

    if num_days_arg != None:
        num_days = num_days_arg
    else:
        num_days = len(test_series)

    print("train_series len: ", len(train_series))

    tf.keras.backend.clear_session()
    model = model_py.def_model(units=units, num_layers=num_layers)
    model.load_weights(filepath=config.MODEL_SAVE_PATH)

    # Reduce the original series
    # forecast_series = train_series[len(train_series) - (num_days + window_size):]
    forecast_series = train_series
    forecast_series = np.array(forecast_series)
    forecast_series = utils.normalise(forecast_series)

    # Use the model to predict data points per window size
    prediction = model.predict(forecast_series[-window_size:][np.newaxis])
    forecast_and_prediction = np.append(forecast_series[:][np.newaxis], np.array([[prediction]]))

    # Predict and add the prediction to the list to use the prediction to make further predictions
    forecasts = [prediction]
    for time in range(num_days - 1):
        print(time)
        prediction = model.predict(forecast_and_prediction[-window_size:][np.newaxis])
        if use_test_keys_for_forcast:
            forecast_and_prediction = np.append(forecast_and_prediction, test_series_norm[time])
        else:
            forecast_and_prediction = np.append(forecast_and_prediction, prediction)
        print("Prediction: ", prediction)
        # print("New list:")
        # print(forecast_and_prediction)
        forecasts.append(prediction)

    # Convert to a numpy array and drop single dimensional axes
    results = np.array(forecasts).squeeze()

    results = utils.inverse_normalise(results)

    print("\nPrediction results:")
    print(results)
    if test_keys_present:
        test_series = test_series[:len(results)]
        print("\nActual values:")
        print(test_series)
        print("\n")

    csvfile = os.path.join(config.TESTING_OUTPUTS_PATH, "prediction.csv")
    with open(csvfile, 'w') as output:
        writer = csv.writer(output, lineterminator='\n')
        for result in results:
            writer.writerow([result])

    # Plot the results
    plt.plot(results)
    if test_keys_present:
        plt.plot(test_series)
    plt.title('Prediction')
    plt.ylabel('Value')
    plt.xlabel('Time')
    plt.legend(['Predicted results', 'Original results'], loc='upper left')
    plt.savefig(os.path.join(config.TESTING_OUTPUTS_PATH, "prediction"))
    plt.show()

    results = utils.normalise(results)
    if test_keys_present:
        test_series = utils.normalise(test_series)

        mse = mean_squared_error(results, test_series)
        mae = np.absolute(np.subtract(test_series, results)).mean()
        print("mse", mse)
        print("mae", mae)