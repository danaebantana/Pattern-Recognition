from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler as standardScaler
import Data as data
import Functions as fun 

scaler = standardScaler()  

MatchFeatures, MatchOutputs = data.loadMLPData()

k = 10
training_sets,testing_sets = fun.k_fold_cross_validation(MatchFeatures,k)
training_outputs,testing_outputs = fun.k_fold_cross_validation(MatchOutputs,k)

MLP = MLPClassifier(solver='sgd', activation='logistic', batch_size=54, learning_rate_init= 0.01, momentum=0.9, 
                    hidden_layer_sizes=(10, 3), random_state=1, max_iter=300)
scores = []

#FOREACH FOLD
for fold in range(k):
    #TRAIN FOR FOLD k
    x_train = training_sets[fold]
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    y_train = training_outputs[fold]
    MLP.fit(x_train, y_train)   #Train MLP classifier
    
    #TEST FOR FOLD k
    x_test = testing_sets[fold]
    x_test = scaler.transform(x_test)
    y_test = testing_outputs[fold]
    y_pred = MLP.predict(x_test) #Test MLP classifier
    score = accuracy_score(y_test, y_pred)  #calculate accuracy
    print('Fold:', fold+1, 'Accuracy:', score)
    scores.append(score)

best_fold = scores.index(max(scores))

print("\nThe most accurate prediction came from fold ",best_fold+1," with prediction accuracy: ",scores[best_fold]*100,"%")
