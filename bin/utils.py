import numpy
from PIL import Image
from sklearn import preprocessing
import lasagne

def get_mean(x_image,  w,  dim,  ImageShape):
    mean_sample = numpy.zeros((w[0],w[1],dim))
    n_samples = (ImageShape[0]-w[0]//2)*(ImageShape[1]-w[1]//2)
    # slide a window across the image
    for x in xrange(w[0]//2, ImageShape[0]-w[0]//2, 1):
        for y in xrange(w[1]//2, ImageShape[1]-w[1]//2, 1):
            window = x_image[(x-w[0]//2):(x+w[0]//2+1), (y-w[1]//2):(y+w[1]//2+1)]
            if window.shape[0] != w[0] or window.shape[1] != w[1]:
                continue
            mean_sample += window.astype(numpy.float_)/n_samples
    return mean_sample

def get_std(x_image,  w,  dim, ImageShape,  mean_sample):
    std_sample = numpy.zeros((w[0],w[1],dim))
    n_samples = (ImageShape[0]-w[0]//2)*(ImageShape[1]-w[1]//2)
    for x in xrange(w[0]//2, ImageShape[0]-w[0]//2, 1):
        for y in xrange(w[1]//2, ImageShape[1]-w[1]//2, 1):
            window = x_image[(x-w[0]//2):(x+w[0]//2+1), (y-w[1]//2):(y+w[1]//2+1)]
            if window.shape[0] != w[0] or window.shape[1] != w[1]:
                continue
            sample = (window.astype(numpy.float_) - mean_sample)**2
            std_sample += sample/n_samples
    return numpy.sqrt(std_sample)

def get_predictions(image,  ImageShape, PatternShape, w,  output_model,  x_mean,  x_std):
    n_batch = 100
    n1 = image.shape[0]
    n2 = image.shape[1]
    diff = (w[0]-1)/2
    valid_windows = n1*n2-diff*2*(n1+n2)+4*diff*diff
    y_preds = numpy.zeros((valid_windows,2))
    c = 0
    # slide a window across the image
    for x in xrange(w[0]//2, image.shape[0]-w[0]//2, 1):
        for y in xrange(w[1]//2, image.shape[1]-w[1]//2, 1):
            window = image[(x-w[0]//2):(x+w[0]//2+1), (y-w[1]//2):(y+w[1]//2+1)]
            if window.shape[0] != w[0] or window.shape[1] != w[1]:
                continue
            sample = window.astype(numpy.float_)
            sample -= x_mean
            sample /= x_std
            sample = sample.reshape(1,sample.size)
            y_preds[c] = output_model(sample.astype('float32'))
            c += 1
    return y_preds

def build_custom_mlp(n_features, n_output, input_var=None, depth=2, width=800, drop_input=.2, drop_hidden=.5):
    network = lasagne.layers.InputLayer(shape=(None, n_features), input_var=input_var)
    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)
    # Hidden layers and dropout:
    #nonlin = lasagne.nonlinearities.very_leaky_rectify
    nonlin = lasagne.nonlinearities.leaky_rectify
    for _ in range(depth):
        network = lasagne.layers.DenseLayer(network, width, nonlinearity=nonlin)
        if drop_hidden:
            network = lasagne.layers.dropout(network, p=drop_hidden)
    # Output layer:
    last_nonlin = lasagne.nonlinearities.softmax
    #last_nonlin = lasagne.nonlinearities.linear
    network = lasagne.layers.DenseLayer(network, n_output, nonlinearity=last_nonlin)
    return network
