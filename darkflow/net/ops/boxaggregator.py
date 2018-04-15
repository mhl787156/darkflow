import tensorflow.contrib.slim as slim
from .baseop import BaseOp
import tensorflow as tf
import numpy as np

class boxaggregator(BaseOp):
    
    def __init__(self, input_image, *args):
        self.input_image = image # BaseOp
        super(boxaggregator, self).__init__(*args)
        
    def forward(self):
        self.out = self.inp.out 
        # TODO write box aggregator

    def speak(self):
        return 'boxaggregator()'
