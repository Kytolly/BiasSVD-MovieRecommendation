from mymodel import Model
from parameter import Parameters
import pandas as pd 
import numpy as np
import tensorflow as tf 

class Trainer():
    def __init__(self):  
        pass
        
    def compile(self, paras: Parameters) :
        self.parameters = paras
        
    def train(self, model:Model, train_set):
        model.fit(train_set)  
        for _ in range(self.parameters.steps):
            model.learn(train_set, self.parameters.lambda_r)   