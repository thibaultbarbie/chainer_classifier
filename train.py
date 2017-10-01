import json
import os
import ast
import numpy as np
from chainer import Variable, iterators, optimizers, training
from chainer.training import extensions
import chainer.links as L

from model import *
from dataset import *

if __name__ == '__main__':

    # Import all the parameters
    with open(os.path.join('.','configuration.json')) as fd:
        json_data = json.load(fd)
    parameters = ast.literal_eval(json.dumps(json_data))

    # Create a new model
    model = L.Classifier(import_model(parameters['model']))

    # Load the parameter of the model if already saved
    if parameters['load_parameters']=="true":
        a=5

    # Import the dataset
    X,Y = import_dataset(parameters['n_positive_data'],parameters['n_negative_data'],
                         parameters['n_parameters'],parameters['dataset_name']) 
    Xtr = X[:parameters['n_train_data']]
    Xval = X[parameters['n_train_data']:]
    Ytr = Y[:parameters['n_train_data']]
    Yval = Y[parameters['n_train_data']:]

    train_iter = iterators.SerialIterator(zip(Xtr,Ytr), batch_size=parameters['batchsize'],
                                          shuffle=True)
    validation_iter = iterators.SerialIterator(zip(Xval,Yval),batch_size=parameters['batchsize'],
                                               repeat=False, shuffle=False)

    model_optimizers = {"SGD" : optimizers.SGD(), "Adam" : optimizers.Adam() }
    optimizer  = model_optimizers[parameters['optimizer']]
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (parameters['epochs'],'epoch'), out='result')

    trainer.extend(extensions.Evaluator(validation_iter, model))
    trainer.extend(extensions.LogReport())    
    trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())
    
    trainer.run()
