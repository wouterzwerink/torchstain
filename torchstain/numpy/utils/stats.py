import numpy as np

def get_mean_std(I):
    mean = np.mean(I)
    return mean, np.std(I, mean=mean)

def standardize(x, mu, std):
    return (x - mu) / std
