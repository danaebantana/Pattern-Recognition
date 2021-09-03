import Functions as fun

Match = []
Match_Results = []
Match_Teams_Result = []
Companies = ["B365","BW","IW","LB"]
Companies_odds = [[]for x in range(4)]
Attributes = []
TeamAttributes= []


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
    
with open("./Data/TeamAttributes.csv","r") as teamAttributesData:
    for line in teamAttributesData:
        Attributes.append(line[:-1].split(","))
del Attributes[0]    
        
#for t in Attributes:
    #TeamAttributes.append([int(t[4]), int(t[5]), int(t[6]), int(t[7]), int(t[8]), int(t[9]), int(t[10]), int(t[11])])
    
