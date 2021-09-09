from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler as standardScaler
import Data as data
import Functions as fun 

#Multi-layer Perceptron is sensitive to feature scaling, so it is highly recommended to scale your data. 
#For example, scale each attribute on the input vector X to [0, 1] or [-1, +1], or standardize it to have mean 0 and variance 1.
scaler = standardScaler()  
#scaler = scaler.fit(data.MatchFeatures)
#MatchFeatures = scaler.transform(data.MatchFeatures)

MatchFeatures = data.MatchFeatures
MatchOutputs = data.MatchFeaturesOutput

k = 10
training_sets,testing_sets = fun.k_fold_cross_validation(MatchFeatures,k)
training_outputs,testing_outputs = fun.k_fold_cross_validation(MatchOutputs,k)

MLP = MLPClassifier(solver='sgd', activation='logistic', learning_rate_init= 0.01, momentum=0.9, hidden_layer_sizes=(12, 3), random_state=1, max_iter=500)

scores = []

#FOREACH FOLD
for fold in range(k):
    #TRAIN FOR FOLD k
    x_train = training_sets[fold]
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    y_train = training_outputs[fold]
    MLP.fit(x_train, y_train)  
    
    #TEST FOR FOLD k
    x_test = testing_sets[fold]
    x_test = scaler.transform(x_test)
    y_test = testing_outputs[fold]
    y_pred = MLP.predict(x_test) 
    score = accuracy_score(y_test, y_pred)
    print('Fold:', fold+1, 'Accuracy:', score)
    scores.append(score)

best_fold = scores.index(max(scores))

print("\nThe most accurate prediction came from fold ",best_fold+1," with prediction accuracy: ",scores[best_fold]*100,"%")
