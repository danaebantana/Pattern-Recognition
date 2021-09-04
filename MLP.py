from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler 
import Data as data 

#Multi-layer Perceptron is sensitive to feature scaling, so it is highly recommended to scale your data. 
#For example, scale each attribute on the input vector X to [0, 1] or [-1, +1], or standardize it to have mean 0 and variance 1.
scaler = StandardScaler()  
MatchFeatures = data.MatchFeatures
MatchOutputs = data.MatchFeaturesOutput

#scaler.fit(X_train)  
#X_train = scaler.transform(X_train)  
#X_test = scaler.transform(X_test)  