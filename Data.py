import Functions as fun

Match = []
Match_Results = []
Match_Teams_Result = []
Companies = ["B365","BW","IW","LB"]
Companies_odds = [[]for x in range(4)]
Attributes = []
TeamAttributes= []
HomeAttributes = []
AwayAttributes = []
MatchFeatures = []
MatchFeaturesOutput = []


with open("./Data/Match.csv","r") as matchData:
    for line in matchData:
        Match.append(line[:-1].split(","))
del Match[0]

num_of_matches = len(Match)
num_of_companies = len(Companies)

for m in Match:
    Gmh = m[9]   #Home Goals
    Gma = m[10]  #Away Goals
    result = fun.match_result(Gmh,Gma)
    Match_Results.append([int(m[6]),result])   #[match_api_id, result]     
    Match_Teams_Result.append([int(m[7]), int(m[8]), result])   #[home_team_api_id, away_team_api_id, result]  

index = 11
for company in range(num_of_companies):
    for match in range(num_of_matches):
        # [match_api_id, name_of_companyH, name_of_companyD, name_of_companyA]
        Companies_odds[company].append([int(Match[match][6]),float(Match[match][index]), float(Match[match][index+1]), float(Match[match][index+2])])
    index += 3

#Data for Neural Network 
with open("./Data/TeamAttributes.csv","r") as teamAttributesData:
    for line in teamAttributesData:
        Attributes.append(line[:-1].split(","))
del Attributes[0]    
        
for t in Attributes:
    TeamAttributes.append([int(t[2]), float(t[4])/50, float(t[6])/50, float(t[9])/50, float(t[11])/50, float(t[13])/50, float(t[16])/50, float(t[18])/50, float(t[20])/50])
    
for m in Match:
    home_team_api_id = int(m[7])
    away_team_api_id = int(m[8])
    for t in TeamAttributes:
        if t[0] == home_team_api_id:
            HomeAttributes = [t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8]]
        if t[0] == away_team_api_id:
            AwayAttributes = [t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8]]
    MatchFeatures.append([HomeAttributes[0], HomeAttributes[1], HomeAttributes[2], HomeAttributes[3], HomeAttributes[4], 
                         HomeAttributes[5], HomeAttributes[6], HomeAttributes[7], AwayAttributes[0], AwayAttributes[1],
                         AwayAttributes[2], AwayAttributes[3], AwayAttributes[4], AwayAttributes[5], AwayAttributes[6], 
                         AwayAttributes[7], float(m[11]), float(m[12]), float(m[13]), float(m[14]), float(m[15]), float(m[16]),
                         float(m[17]), float(m[18]), float(m[19]), float(m[20]), float(m[21]), float(m[22])])         
    home_goals = m[9]   #Home Goals
    away_goals = m[10]  #Away Goals
    result = fun.match_result(home_goals,away_goals)
    output = fun.cal_output(result)
    MatchFeaturesOutput.append(output)
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
