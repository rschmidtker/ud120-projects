import math

import scipy.stats
print (scipy.stats.entropy([2,2],base=2))

math.log2(2)



""" quiz materials for feature scaling clustering """

### FYI, the most straightforward implementation might 
### throw a divide-by-zero error, if the min and max
### values are the same
### but think about this for a second--that means that every
### data point has the same value for that feature!  
### why would you rescale it?  Or even use it at all?
def featureScaling(arr):
    xmin = min(arr)
    xmax = max(arr)

    ret = []
    for i in arr:
        if xmin != xmax:
            x = i - xmin / xmax - xmin
        else:
            x = 0.5
        ret.append(x)
        
    return ret

# tests of your feature scaler--line below is input data
data = [115., 140., 175.]
print (featureScaling(data))

import numpy
from sklearn.preprocessing import MinMaxScaler
weigths = numpy.array([[115.], [140.], [175.]])
scaler = MinMaxScaler()
rescaled_weights = scaler.fit_transform(weigths)
print(rescaled_weights)






