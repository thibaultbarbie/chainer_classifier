import numpy as np
import chainer
from chainer import Chain
import chainer.functions as F
import chainer.links as L

from models.mlp import *


def import_model(model_name):
    all_models={"MLP1" : MLP(50,2),"MLP2" : MLP(10,2)}
    return all_models[model_name]
