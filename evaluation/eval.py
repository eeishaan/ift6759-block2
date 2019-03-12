from horoma.test import test
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
MODULE_DIR = os.path.realpath(os.path.join(CURR_DIR, '../'))
sys.path.insert(0, MODULE_DIR)


def eval_model(model_path, dataset_dir, split):
    '''
    # MODIFY HERE #
    This function is meant to be an example

    '''

    # # SETUP DATASET # #
    # Load requested dataset
    """ IMPORTANT # of example per splits.
    "train" = 1614216
    "valid" = 201778
    "test"  = 201778

    Files available the test folder:
    definition_labels.txt
    train_x.dat
    valid_y.txt
    valid_x.dat
    test_x.dat
    test_y.txt

    You need to load the right one according to the `split`.
    """
    args = SimpleNamespace(
        cluster='kmeans',
        embedding='cae',
        data_dir=dataset_dir,
        model_path=model_path,
        split=split
    )
    result = test(args)
    return result


if __name__ == "__main__":

    # Put your group name here
    group_name = "b2phot3"

    model_path = None
    # model_path should be the absolute path on shared disk to your best model.
    # You need to ensure that they are available to evaluators on Helios.

    #########################
    # DO NOT MODIFY - BEGIN #
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_dir", type=str, default="/rap/jvb-000-aa/COURS2019/etudiants/data/horoma/",
                        help="Absolute path to the dataset directory.")
    parser.add_argument("-s", "--dataset_split", type=str, choices=[
                        'valid', 'test', 'train'], default="valid", help="Which split of the dataset should be loaded from `dataset_dir`.")
    parser.add_argument("-r", "--results_dir", type=str, default="./",
                        help="Absolute path to where the predictions will be saved.")
    args = parser.parse_args()

    # Arguments validation
    if group_name == "b1photN":
        print("'group_name' is not set.\nExiting ...")
        exit(1)

    if model_path is None or not os.path.exists(model_path):
        print("'model_path' ({}) does not exists or unreachable.\nExiting ...".format(
            model_path))
        exit(1)

    if args.dataset_dir is None or not os.path.exists(args.dataset_dir):
        print("'dataset_dir' does not exists or unreachable..\nExiting ...")
        exit(1)

    y_pred = eval_model(model_path, args.dataset_dir, args.dataset_split)

    assert type(y_pred) is np.ndarray, "Return a numpy array"
    assert len(y_pred.shape) == 1, "Make sure ndim=1 for y_pred"

    results_fname = os.path.join(
        args.results_dir, "{}_pred_{}.txt".format(group_name, args.dataset_split))

    print('\nSaving results to ({})'.format(results_fname))
    np.savetxt(results_fname, y_pred, fmt='%s')
    # DO NOT MODIFY - END #
    #######################
