import os
import numpy as np
import pandas as pd
from keras import layers, models

class ModelNotFound:
    pass

def get_row(year, day, hour):
    meta = pd.read_csv(os.path.join('indexes_data', 'meta.csv'))
    return meta.query("YEAR == @year and DOY == @day and UT == @hour").iloc[0]

def get_model(name):
    inp = None
    if name == 'NN-f107':
        inp = layers.Input((12,))
    elif name == 'NN':
        inp = layers.Input((18,))
    else:
        raise ModelNotFound
    x = layers.Dense(512, activation='relu')(inp)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dense(5112, activation='relu')(x)
    model = models.Model(inp, x)

    if name == 'NN-f107':
        model.load_weights(os.path.join('weights', 'NN-f107.h5'))
    elif name == 'NN':
        model.load_weights(os.path.join('weights', 'NN.h5'))

    class Model:

        def __init__(self, fields, model):
            self.fields = fields
            self.model = model

        def __call__(self, year, day, hour):
            row = get_row(year, day, hour)
            x = row[self.fields]
            pred = self.model.predict(np.expand_dims(x.values, axis=0))
            return pred[0].reshape(72, 71)

    fields_for_models = {
       'NN': ['YEAR', 'DOY', 'UT', 'Kp', 'R', 'ap', 'f10_7', 'AE', 'AL', 'AU',
                'f107_ma05', 'f107_ma11', 'f107_ma81', 'f107_sd05', 'f107_sd11',
                'f107_sd81', 'COY', 'SOY'],
       'NN-f107': ['YEAR', 'DOY', 'UT', 'f10_7', 'f107_ma05', 'f107_ma11', 'f107_ma81', 'f107_sd05', 'f107_sd11',
                    'f107_sd81', 'COY', 'SOY']
    }

    return Model(fields_for_models[name], model)
