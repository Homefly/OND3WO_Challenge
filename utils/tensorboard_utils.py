""" tensorboard_utils

Tensorboard specific helpers

"""

from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint


def make_callbacks(model_name):
    """ Setup callbacks for keras model training

    :param model_name: name of model [str]
    :return: [callback]
    """
    checkpoint_path = "model/model_" + model_name + ".best.hdf5"

    early_stop = EarlyStopping(monitor="val_acc", min_delta=0, patience=20,
                               verbose=0, mode="auto")
    tensorboard_callback = TensorBoard(log_dir="model/logs/")
    model_checkpoint = ModelCheckpoint(
        checkpoint_path,
        #"model/model-ffNN-{epoch:02d}-{acc:.4f}.hdf5",
        monitor="val_acc", verbose=1, save_best_only=True, mode="auto")
    callbacks_list = [tensorboard_callback, model_checkpoint, early_stop]

    return callbacks_list