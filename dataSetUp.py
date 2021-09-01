import functions as fun

Match = []
Match_Results = []
companies_odds = [[]for x in range(4)]
companies = ["B365","BW","IW","LB"]

with open("./kaggle_soccer_csv_matlab/Match.csv","r") as match_csv:
    for line in match_csv:
        Match.append(line[:-1].split(","))



num_of_matches = len(Match)
num_of_companies = len(companies_odds)

for m in Match:
    home_goals = m[10]
    away_goals = m[11]
    result = fun.match_result(home_goals,away_goals)
    Match_Results.append([int(m[6]),result])            #Match_Results stores the match result for each match_id. Match_Results[match,result]

index = 11
for company in range(num_of_companies):
    for m in range(num_of_matches):
        companies_odds[company].append([int(Match[m][6]),float(Match[m][index]), float(Match[m][index+1]), float(Match[m][index+2])])
    index += 3