## Train the model

from random import sample as random_s
# cross-validation
def get_train_test_val(X_all, y_all, SY, EY):
    test_year = [1988,1998,2006,2018,2020]     # select data only for testing and final model evaluation
    NY_train = 25                              # number of years for training
    test_X = X_all.sel(time = X_all.time.dt.year.isin([test_year]))
    test_y = y_all.sel(time = y_all.time.dt.year.isin([test_year]))
    all_year = np.arange(SY,EY+1)
    remain_year = set(all_year) - set(test_year)
    train_year = random_s(remain_year, NY_train)
    train_X = X_all.sel(time = X_all.time.dt.year.isin([train_year]))
    train_y = y_all.sel(time = y_all.time.dt.year.isin([train_year]))
    val_year = set(remain_year) - set(train_year)
    val_X = X_all.sel(time = X_all.time.dt.year.isin([list(val_year)]))
    val_y = y_all.sel(time = y_all.time.dt.year.isin([list(val_year)]))
    return train_X, train_y, val_X, val_y, test_X, test_y

# train the model
import json
import pickle
def train_model(model, train_X, train_y, val_X, val_y, callbacks_path, epochs, batch_size):
    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath=callbacks_path,
            monitor='val_loss',   # tf.keras.metrics.AUC(from_logits=True)
            save_best_only=True,
        )
    ]
    history = model.fit(
        train_X, train_y,
        batch_size=batch_size,
        epochs=epochs,
        verbose=True,
        shuffle=True,
        validation_data=(val_X, val_y),
        callbacks=callbacks_list
    )
    history = history.history
    return history

# Save History (pickle)
def save_history(history_path):
    with open(history_path, "wb") as f:
        pickle.dump(history, f)

# Load History
def load_history(hist_path):
    history = pickle.load(open(history_path), "rb")
    return history