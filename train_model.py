# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 06:20:03 2022

@author: sanaalamgeer
"""
import pandas as pd
import numpy as np

import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import GCNSupervisedGraphClassification
from stellargraph import StellarGraph

from stellargraph import datasets

from sklearn import model_selection
from IPython.display import display, HTML

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import binary_crossentropy, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt

from load_data_custom import *

from scipy.stats import spearmanr, kendalltau, pearsonr
import time
####################################################
#%%
def root_mean_squared_error(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def create_graph_classification_model(generator):
    gc_model = GCNSupervisedGraphClassification(
        layer_sizes=[32, 32, 64, 64, 128],
        activations=["elu", "elu", "elu", "elu", "elu"],
        generator=generator,
        dropout=0.5,
    )
    x_inp, x_out = gc_model.in_out_tensors()
    predictions = Dense(units=512, activation="elu")(x_out)
    predictions = Dense(units=512, activation="elu")(predictions)
    predictions = Dense(units=1)(predictions)

    # Let's create the Keras model and prepare it for training
    model = Model(inputs=x_inp, outputs=predictions)
    #opt = Adam(0.005)
    opt = SGD(learning_rate=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss=mean_squared_error, metrics=["mse"])
    model.summary()
    return model

def train_fold(model, train_gen, test_gen, es, mc, epochs):
    history = model.fit(
        train_gen, epochs=epochs, validation_data=test_gen, verbose=1, callbacks=[es, mc], shuffle=True
    )
    
    # calculate performance on the test data and return along with history
    test_metrics = model.evaluate(test_gen, verbose=1)
    test_acc = test_metrics[model.metrics_names.index("mse")]
	
    predictions = model.predict(test_gen)
    targ = graph_labels.iloc[test_index].values
    print(targ.shape, predictions.shape)
    #targ = targ.reshape(targ.shape[0], 1)
    plcc = abs(pearsonr(targ, predictions[:, 0])[0])
    srcc = abs(spearmanr(targ, predictions[:, 0])[0])
    krcc =  abs(kendalltau(targ, predictions[:, 0])[0])
    rmse = root_mean_squared_error(targ, predictions[:, 0])
    
    print('PLCC:', plcc)
    print('SROCC:', srcc)
    print('KRCC:', krcc)
    print('RMSE:', rmse) 

    return history, test_acc, plcc, srcc, krcc, rmse

def get_generators(train_index, test_index, graph_labels, batch_size):
    train_gen = generator.flow(
        train_index, targets=graph_labels.iloc[train_index].values, batch_size=batch_size
    )
    test_gen = generator.flow(
        test_index, targets=graph_labels.iloc[test_index].values, batch_size=batch_size
    )
    return train_gen, test_gen
####################################################
#%%
start_time = time.time()
print("Loading dataset...")
graphs, graph_labels = get_graphs_data("variables/lfdd/")
print("Done!!!")
print("--- %s  dataset loaded in seconds ---" % (time.time() - start_time))

print(graphs[0].info())
print(graph_labels[0])
####################################################
generator = PaddedGraphGenerator(graphs=graphs)
####################################################
model = create_graph_classification_model(generator)
####################################################
#%%
epochs = 200  # maximum number of training epochs
folds = 10  # the number of folds for k-fold cross validation
n_repeats = 2  # the number of repeats for repeated k-fold cross validation
####################################################
es = EarlyStopping(
    monitor="val_loss", min_delta=0, patience=100, restore_best_weights=True
)
mc = ModelCheckpoint('checkpoint/gcn_lfdd.hdf5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
####################################################
#%%
test_mse = []
sr, pl, kr, rmse = [], [], [], []

stratified_folds = model_selection.RepeatedKFold(
    n_splits=folds, n_repeats=n_repeats, random_state=1
).split(graph_labels, graph_labels)
####################################################
#%%
train_ptr = open('train_pointers.txt', 'wt')

start_time = time.time()
for i, (train_index, test_index) in enumerate(stratified_folds):
    print(f"Training and evaluating on fold {i+1} out of {folds * n_repeats}...")
    train_gen, test_gen = get_generators(train_index, test_index, graph_labels, batch_size=30)

    history, mse, plcc, srcc, krcc, rse = train_fold(model, train_gen, test_gen, es, mc, epochs)
    sr.append(srcc)
    pl.append(plcc)
    kr.append(krcc)
    rmse.append(rse)
    
    train_ptr.write('{} {} {} {}\n'.format(plcc, srcc, krcc, rse))

    test_mse.append(mse)
    
print("--- %s Training complete in seconds ---" % (time.time() - start_time))
####################################################   
train_ptr.close()

####################################################
print(
    f"Accuracy over all folds mean: {np.mean(test_mse)*100:.3}% and std: {np.std(test_mse)*100:.2}%"
)
print(
    f"Mean correlations: srcc:{np.mean(sr)}, plcc:{np.mean(pl)}, krcc:{np.mean(kr)} and rmse:{np.mean(rse)}"
)
#####################################################
