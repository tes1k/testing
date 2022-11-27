import traceback

import source.model as model
import source.forecast as forecast
import source.load_data as load_data
import source.utils as utils

ch = int(input("Run the program in this order of 1 to 4. Progress will be saved in each step.\n"
               "1: Set choices(csv details, model choice)\n"
           "2: Tune hyperparameters\n"
           "3: Train model\n"
           "4: forecast\n"
           "Enter your choice: "))
if ch == 1:
    utils.read_csv_store_user_choice()
elif ch == 2:
    ch = input("Do you wish to continue previous tuning (y/n): ")
    model.tune_hyperparameters(continue_training=True) if (ch.lower() == 'y') else model.tune_hyperparameters(continue_training=False)
elif ch == 3:
    model.train_model()
elif ch == 4:
    try:
        test_set_len = len(load_data.load_test_from_file())
        print(f"\n--Test set length: {test_set_len}\n")
    except:
        traceback.print_exc()
        print("Warning: No test set found")
        pass
    ch_ = int(input(""
                    "1) Forecast for x days by predicting (t+1) only. (This will be done by appending the input list with values from test set, so that only the next value is unknown)\n"
                    "2) Forecast for x days without appending values from test set\n"
                    "3) Forecast for x days without a test set\n"
                    "Enter your choice: "))
    num_days = int(input("Enter the number of days you wish to forecast: "))
    if ch_ == 1:
        num_days = min(test_set_len, num_days)
        forecast.forecast(test_keys_present=True, use_test_keys_for_forcast=True, num_days_arg=num_days)
    elif ch_ == 2:
        num_days = min(test_set_len, num_days)
        forecast.forecast(test_keys_present=True, use_test_keys_for_forcast=False, num_days_arg=num_days)
    elif ch_ == 3:
        forecast.forecast(test_keys_present=False, use_test_keys_for_forcast=False, num_days_arg=num_days)
