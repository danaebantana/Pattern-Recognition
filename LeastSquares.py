import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import Functions as fun
import Data as data

# Get Data
numberOfCompanies,companies_odds,Match_Results,companies,matches = data.loadData()

# Calculating weights through ls algorithm

k = 10    #10-fold-cross-validation
company_fold_weight = [[]for x in range(numberOfCompanies)]
training_set = [[]for x in range(numberOfCompanies)]
testing_set = [[]for x in range(numberOfCompanies)]

#Foreach company 'B365', 'BW', 'IW' & 'LB'
for company in range(numberOfCompanies):
    training_set_i,testing_set_i = fun.k_fold_cross_validation(companies_odds[company],k)
    num_training_set_matches = len(training_set_i[company])
    fold_num_of_matches = len(testing_set_i[company])
    num_of_matches = num_training_set_matches + fold_num_of_matches
    
    #Foreach fold k=10
    for fold in range(k):
        X = fun.modify_X(training_set_i[fold])  #set x0=1
        fold_weights = [[]for x in range(3)]
        for i,class_i in enumerate(["H","D","A"]):
            d = fun.desired_values(fold, fold_num_of_matches, Match_Results, class_i, num_of_matches, -1)
            w_i = np.linalg.inv(X.T @ X) @ (X.T @ d)
            fold_weights[i] = w_i
        company_fold_weight[company].append((fold_weights))
    training_set[company] = (training_set_i)
    testing_set[company] = (testing_set_i)

# Testing the classifier with the calculated weights
scores = [[]for x in range(numberOfCompanies)]
W = []

for company in range(numberOfCompanies):
    for fold in range(k):
        results = fun.score_classifier(testing_set[company][fold],company_fold_weight[company][fold],fold,Match_Results)
        scores[company].append(results)  #results= [correct, wrong] for each betting company.

#Evaluating best weights and calculating the classification accuracy
for company,company_score in enumerate(scores):
    max_score = company_score[0]
    best_fold = 0
    for fold in range(1,k):
        if(company_score[fold] > max_score):
            max_score = company_score[fold]
            best_fold = fold
    best_score = (max_score/fold_num_of_matches)*100   #Score of the best fold of a company 
    W.append(company_fold_weight[company][best_fold])  #Storing weights of the best fold for each betting company
    scores[company] = [best_fold,int(best_score)]
    
max_score = scores[0][1]
best_fold = scores[0][0]
best_company = 0
for company,company_score in enumerate(scores):
    if(company_score[1] > max_score):
            max_score = company_score[1]
            best_fold = company_score[0]
            best_company = company
            
for company,company_score in enumerate(scores):
    print("Company:", companies[company], "Accuracy:", company_score[1], "%")
print("\nBest betting company: ", companies[best_company])
print("Best fold: ", best_fold)
print("Best percentage: ", max_score, "%\n")


#Plotting and Visualization of results
xx, yy = np.meshgrid(range(10), range(10))
names = ["HOME","DRAW","AWAY"]
colors = ["lightblue","orange","red"]

for c,company in enumerate(W):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print("Best weights for company: ",companies[c])
    for j,plane in enumerate(company):
        print("For ",names[j],": ",plane)
        z = (-plane[1]*xx - plane[2]*yy - plane[0])/plane[3]
        ax.plot_surface(xx, yy, z, color = colors[j],alpha = 0.5)
    print("\n")

    to_scatter_home_wins = []
    to_scatter_draw = []
    to_scatter_away_wins = []

    for i in range(num_of_matches):
        if(data.Match_Results[i][1] == "H"):
            to_scatter_home_wins.append([companies_odds[c][i][1],companies_odds[c][i][2],companies_odds[c][i][3]])
        elif(data.Match_Results[i][1] == "D"):
            to_scatter_draw.append([companies_odds[c][i][1],companies_odds[c][i][2],companies_odds[c][i][3]])
        else:
            to_scatter_away_wins.append([companies_odds[c][i][1],companies_odds[c][i][2],companies_odds[c][i][3]])

    to_scatter_home_wins = np.array(to_scatter_home_wins)
    to_scatter_draw = np.array(to_scatter_draw)
    to_scatter_away_wins = np.array(to_scatter_away_wins)

    ax.scatter(to_scatter_home_wins[:,0],to_scatter_home_wins[:,1],to_scatter_home_wins[:,2],color = colors[0],label = "Home Wins")
    ax.scatter(to_scatter_draw[:,0],to_scatter_draw[:,1],to_scatter_draw[:,2],color = colors[1], label = "Draw")
    ax.scatter(to_scatter_away_wins[:,0],to_scatter_away_wins[:,1],to_scatter_away_wins[:,2],color = colors[2], label = "Away Wins")

    plt.title("'Least Squares' for : "+ companies[c]+
    "\n Evaluation score: "+str(scores[c][1])+"%, achieved from fold: "+str(scores[c][0])+"/"+str(k))
    ax.set_xlabel("HOME WINS")
    ax.set_ylabel("DRAW")
    ax.set_zlabel("AWAY WINS")
    plt.legend()
    plt.tight_layout()

plt.show()
