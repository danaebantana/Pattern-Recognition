import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import Functions as fun
import Data as data


# CALCULATING WEIGHTS <>
k_fold = 10
company_fold_weight = [[]for x in range(data.num_of_companies)]
training_set = [[]for x in range(data.num_of_companies)]
testing_set = [[]for x in range(data.num_of_companies)]
#FOREACH COMPANY
for company in range(data.num_of_companies):
    training_set_i,testing_set_i = fun.k_fold_cross_validation(data.Companies_odds[company],k_fold)
    training_set_matches = len(training_set_i[company])
    testing_set_matches = len(testing_set_i[company])
    num_of_matches = training_set_matches + testing_set_matches
    
    #FOREACH FOLD
    for k in range(k_fold):
        X = fun.featureMatrix(training_set_i[k])   #set x0 

        fold_weights = [[]for x in range(3)]
        for i,class_i in enumerate(["H","D","A"]):  #px. i=0 class_i=H
            y = fun.observedValue(data.Match_Results,class_i,k,testing_set_matches,num_of_matches)
            w_i = fun.lms_Robbins_Monro(X,y)
            fold_weights[i] = w_i
        company_fold_weight[company].append((fold_weights))
    training_set[company] = (training_set_i)
    testing_set[company] = (testing_set_i)
# CALCULATING WEIGHTS </>


# TESTING SET AND EVALUATING BEST WEIGHTS <>
scores = [[]for x in range(data.num_of_companies)]
W = []

for company in range(data.num_of_companies):
    for k in range(k_fold):
        results = fun.score_weights(testing_set[company][k],company_fold_weight[company][k],k,data.Match_Results)
        scores[company].append(results)  #results= [correct, wrong] for each betting company.

for company,company_score in enumerate(scores):
    max_score = company_score[0][0]
    best_fold = 0
    for fold in range(1,k_fold):
        if(company_score[fold][0] > max_score):
            max_score = company_score[fold][0]
            best_fold = fold
    best_score = (max_score/testing_set_matches)*100                    #Score of the best fold in % percentage (correct_guess/total_matches)
    W.append(company_fold_weight[company][best_fold])                       #Storing weights of the best fold for each betting company
    scores[company] = [best_fold,int(best_score)]
    
max_score = scores[0][1]
best_fold = scores[0][0]
best_company = 0
for company,company_score in enumerate(scores):
    if(company_score[1] > max_score):
            max_score = company_score[1]
            best_fold = company_score[0]
            best_company = company
print("Best betting company: ", data.Companies[best_company])
print("Best fold: ", best_fold)
print("Best percentage: ", max_score)
# TESTING SET AND EVALUATING BEST WEIGHTS </>




# FOR PLOTTING PURPOSES <>

xx, yy = np.meshgrid(range(10), range(10))
names = ["HOME","DRAW","AWAY"]
colors = ["lightblue","orange","red"]

for c,company in enumerate(W):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print("Best weights for company: ",data.Companies[c])
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
            to_scatter_home_wins.append([data.Companies_odds[c][i][1],data.Companies_odds[c][i][2],data.Companies_odds[c][i][3]])
        elif(data.Match_Results[i][1] == "D"):
            to_scatter_draw.append([data.Companies_odds[c][i][1],data.Companies_odds[c][i][2],data.Companies_odds[c][i][3]])
        else:
            to_scatter_away_wins.append([data.Companies_odds[c][i][1],data.Companies_odds[c][i][2],data.Companies_odds[c][i][3]])


    to_scatter_home_wins = np.array(to_scatter_home_wins)
    to_scatter_draw = np.array(to_scatter_draw)
    to_scatter_away_wins = np.array(to_scatter_away_wins)



    ax.scatter(to_scatter_home_wins[:,0],to_scatter_home_wins[:,1],to_scatter_home_wins[:,2],color = colors[0],label = "Home Wins")
    ax.scatter(to_scatter_draw[:,0],to_scatter_draw[:,1],to_scatter_draw[:,2],color = colors[1], label = "Draw")
    ax.scatter(to_scatter_away_wins[:,0],to_scatter_away_wins[:,1],to_scatter_away_wins[:,2],color = colors[2], label = "Away Wins")


    plt.title("Odds and Best Decision Boundaries for : "+data.Companies[c]+
    "\n Evaluation score: "+str(scores[c][1])+"%, achieved from fold: "+str(scores[c][0])+"/"+str(k))
    ax.set_xlabel("HOME WINS")
    ax.set_ylabel("DRAW")
    ax.set_zlabel("AWAY WINS")
    #ax.legend()
    plt.legend()
    plt.tight_layout()

plt.show()

# FOR PLOTTING PURPOSES </>

