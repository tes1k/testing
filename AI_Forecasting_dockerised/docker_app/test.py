# IMPORTANT: "my_model.h5" and "min_max_scalar_data.npy" from save_files folder are needed to make predictions

import source.forecast as forecast
import source.load_data as data
import traceback

try:
    test_set_len = len(data.load_test_from_file())
    print(f"\n--Test set length: {test_set_len}\n")
except:
    traceback.print_exc()
    print("-----------WARNING: No test set found")
    raise

num_days = test_set_len  # To set a custom forecast length, change this number

# Forecast for x days by predicting (t+1) only. (This will be done by appending the input list with values from test set, so that only the next value is unknown)
forecast.forecast(test_keys_present=True, use_test_keys_for_forcast=True, num_days_arg=num_days)

# Forecast for x days without appending values from test set
# forecast.forecast(test_keys_present=True, use_test_keys_for_forcast=False, num_days_arg=num_days)

# Forecast for x days without a test set
# forecast.forecast(test_keys_present=False, use_test_keys_for_forcast=False, num_days_arg=num_days)
