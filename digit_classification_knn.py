from sklearn.datasets import load_digits
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier

digits = load_digits()
x_train = digits.data
y_train = digits.target


classifier = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
classifier.fit(x_train, y_train)


two = [  	0.,   0.,  11.,  16.,  0.,   10.,   0.,   0.,
          	0.,   5.,  16.,  12.,  11.,  12.,   0.,   0.,
          	0.,   3.,  13.,    1.,    5.,  15.,   0.,   0.,
          	0.,   0.,   0.,     0.,  12.,  11.,   0.,   0., 
          	0.,   0.,   0.,     1.,  16.,   7.,   0.,   0., 
          	0.,   0.,   0.,   10.,  15.,   0.,   0.,   0.,
          	0.,   0.,  12.,  16.,  16.,  11.,   1.,   0.,
          	0.,   0.,  16.,  16.,   8.,  13.,  16.,   8.]

np_two = np.array(two)

prediction = classifier.predict(np_two.reshape(1,-1))
print("predicted value ", list(prediction))



three = [  	0.,   0.,  11.,  16.,  0.,   10.,   0.,   0.,
          	0.,   5.,  16.,  12.,  11.,  12.,   0.,   0.,
               	0.,   3.,   0.,    1.,    5.,  15.,   0.,   0.,
          	0.,   0.,   0.,     0.,  12.,  11.,   0.,   0., 
          	0.,   0.,   0.,     1.,  16.,   7.,   0.,   0., 
          	0.,   0.,   0.,   0.,  15.,   10.,   0.,   0.,
          	0.,   0.,   0.,   0.,  0.,  11.,    16.,   0.,
          	0.,   0.,  16.,  16.,   8.,  13.,  0.,   0.]

np_three = np.array(three)

prediction = classifier.predict(np_three.reshape(1,-1))
print("predicted value ", list(prediction))


