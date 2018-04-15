from .simple import *
from .convolution import *
from .boxaggregator import *
from .baseop import HEADER, LINE

op_types = {
	'convolutional': convolutional,
	'conv-select': conv_select,
	'connected': connected,
	'maxpool': maxpool,
	'leaky': leaky,
	'dropout': dropout,
	'flatten': flatten,
	'avgpool': avgpool,
	'softmax': softmax,
	'identity': identity,
	'crop': crop,
	'local': local,
	'select': select,
	'route': route,
	'reorg': reorg,
	'conv-extract': conv_extract,
	'extract': extract,
	'boxaggregator': boxaggregator
}

def op_create(inpt, *args):
	layer_type = list(args)[0].type
	if layer_type == 'boxaggregator':
		op_types[layer_type](identity(inpt),*args)
	return op_types[layer_type](*args)