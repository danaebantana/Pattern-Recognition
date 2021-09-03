import numpy as np

def match_result(Gmh,Gma):
    diff_goal = int(Gmh) - int(Gma)
    if(diff_goal > 0):
        return "H"
    elif(diff_goal == 0):
        return "D"
    else:
        return "A"
    
def k_fold_cross_validation(inputs,k_fold):
    matches = len(inputs)
    fold = int(matches/k_fold)   #number of matches in each fold.
    totalMatches = fold*k_fold   #total number of matches for 10 folds.
    testing_set = [] 
    training_set = []
    for k in range(k_fold):   #fold=0 to 9
        start_test_point = k*fold   #0, 2259, 4518 ...
        for m in range(0,totalMatches,fold):  # for each_fold 
            if(m == start_test_point):
                testing_set.append(inputs[m:m+fold]) #matches of each fold
            else:
                if(len(training_set) < k+1):
                    training_set.append(inputs[m:m+fold])
                else:
                    training_set[k] += inputs[m:m+fold]
    return training_set,testing_set

def featureMatrix(company):
    X = []
    for i in range(len(company)):   #company[i][0] stores the match_id.
        home_odd = company[i][1]    #company[i][1] stores odds for home win
        draw_odd = company[i][2]    #company[i][2] stores odds for draw
        away_odd = company[i][3]    #company[i][3] stores odds for away win
        X.append([home_odd,draw_odd,away_odd])
    X = np.array(X)
    X = np.insert(X,0,1.0,axis=1)   # X = [ 1.0 home_odd draw_odd away_odd]. Added x0 = 1
    return X

def observedValue(results,class_i,fold,each_fold,total_matches):
    print(fold*each_fold, (fold+1)*each_fold)
    y = []
    for i in range(0,fold*each_fold):
        if(class_i == results[i][1]):
            y.append(1)
        else:
            y.append(0)
    for i in range((fold+1)*each_fold,total_matches):
        if(class_i == results[i][1]):
            y.append(1)
        else:
            y.append(0)
    return y

def lms(X,y):
    def hypothesis(match,W):
        result = 0
        for i in range(len(W)):
            result += W[i]*X[match][i]
        return result
    init_weights = [0.0,0.0,0.0,0.0]
    weights = [0.0,0.0,0.0,0.0]
    num_of_matches = len(X)
    a = 1
    convergence = 0.0000001
    for match in range(1,num_of_matches):
        if(a <= convergence):
            break
        else:
            learning_rate = a/match #+0.001 bit more accurate results
            for w_i in range(4):
                Error = y[match]-hypothesis(match,init_weights)
                step = learning_rate*Error
                weights[w_i] = init_weights[w_i]+step
            init_weights = weights
    return weights

