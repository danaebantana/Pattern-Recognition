import numpy as np

def match_result(home_goals,away_goals):
    diff_goal = int(home_goals) - int(away_goals)
    if(diff_goal > 0):
        return "H"
    elif(diff_goal == 0):
        return "D"
    else:
        return "A"
    
def k_fold_cross_validation(inputs,k):
    matches = len(inputs)
    matches_fold = int(matches/k)   #number of matches in each fold.
    totalMatches = matches_fold*k   #total number of matches for 10 folds.
    testing_set = [] 
    training_set = []
    for fold in range(k):   #fold=0 to 9
        start_test_point = fold*matches_fold   #0, 2259, 4518 ...
        for m in range(0,totalMatches,matches_fold):  # for each_fold 
            if(m == start_test_point):
                testing_set.append(inputs[m:m+matches_fold]) #testing matches of each fold
            else:
                if(len(training_set) < fold+1):
                    training_set.append(inputs[m:m+matches_fold])
                else:
                    training_set[fold] += inputs[m:m+matches_fold]
    return training_set,testing_set

def modify_X(company_odds):
    X = []
    for i in range(len(company_odds)):   #company[i][0] stores the match_id.
        home_odd = company_odds[i][1]    #company[i][1] stores odds for home win
        draw_odd = company_odds[i][2]    #company[i][2] stores odds for draw
        away_odd = company_odds[i][3]    #company[i][3] stores odds for away win
        X.append([home_odd,draw_odd,away_odd])
    X = np.array(X)
    X = np.insert(X,0,1.0,axis=1)   # X = [ 1.0 home_odd draw_odd away_odd]. Added x0 = 1
    return X

def desired_values(fold,each_fold,results,class_i,total_matches,not_class_i_symbol):
    d = []
    for i in range(0,fold*each_fold):
        if(class_i == results[i][1]): #results[i][1] = [match_id, 'H' or 'D' or 'A']  
            d.append(1)
        else:
            d.append(not_class_i_symbol)  #0 for robbins-monro, -1 for ls
    for i in range((fold+1)*each_fold,total_matches):
        if(class_i == results[i][1]):
            d.append(1)
        else:
            d.append(not_class_i_symbol)
    return d

def y_output(X,match,W):
    y = 0
    for i in range(len(W)):
        y += W[i]*X[match][i]
    return y

def lms_robbins_monro(X,d):
    init_weights = [0.0,0.0,0.0,0.0]
    weights = [0.0,0.0,0.0,0.0]
    a = 1
    convergence = 0.0000001
    for match in range(1,len(X)):
        if(a <= convergence):
            break
        else:
            a_i = a/match
            for w_i in range(4):
                J = d[match] - y_output(X,match,init_weights)  #( d(xᵢ)-y(xᵢ) ) ^ 2
                step = a_i * J
                weights[w_i] = init_weights[w_i] + step
            init_weights = weights
    return weights

def score_classifier(data,W,fold,results):
    outcome = ["H","D","A"]
    correct = 0
    num_data = len(data)  #2259
    start = fold*num_data
    end = start+num_data
    for m in range(start,end):
        result = []
        k = m-start
        for i in range(len(W)):  #len(W) = 3
            y = W[i][0] + W[i][1]*data[k][1] + W[i][2]*data[k][2] + W[i][3]*data[k][3]  #classifier output
            result.append((1-y)**2)
        best_fit = result.index(min(result))    #position of min result
        if(outcome[best_fit] == results[m][1]):
            correct += 1
    return correct

def cal_output(output):
    if output == "H":
        return [1,0,0]
    elif output == "D":
        return [0,1,0]
    elif output == "A":
        return [0,0,1]
    

      