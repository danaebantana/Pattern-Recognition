def match_result(Gmh,Gma):
    diff_goal = int(Gmh) - int(Gma)
    if(diff_goal > 0):
        return "H"
    elif(diff_goal == 0):
        return "D"
    else:
        return "A"