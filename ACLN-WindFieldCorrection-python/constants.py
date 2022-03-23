import os
import numpy as np

def get_dir(directory):
    """
    Creates the given directory if it does not exist.

    @param directory: The path to the directory.
    @return: The path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

save_name = '2109121827'
save_dir = '../save/'
train_save_dir = get_dir(os.path.join(save_dir, 'train/', save_name))
test_save_dir = get_dir(os.path.join(save_dir, 'test/', save_name))
save_models_dir = get_dir(os.path.join(save_dir, 'models/', save_name))
loss_save_dir = get_dir(os.path.join(save_dir,'loss/', save_name))

train_original_dir = '../WindData_40N110E_10N140E_0125_2018/train/ForecastData/'
train_revised_dir = '../WindData_40N110E_10N140E_0125_2018/train/ReanalysisData/'
test_original_dir = '../WindData_40N110E_10N140E_0125_2018/test/ForecastData/'
test_revised_dir = '../WindData_40N110E_10N140E_0125_2018/test/ReanalysisData/'

data_height = 241
data_width = 241
