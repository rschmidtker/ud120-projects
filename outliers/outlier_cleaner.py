#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []


    ### your code goes here
    from operator import itemgetter
    for i in range(len(predictions)):
        cleaned_data.append((ages[i], net_worths[i], abs(predictions[i] - net_worths[i])))
    tmp = cleaned_data
    tmp = sorted(tmp, key=itemgetter(2))
    a,b,level = tmp[int(len(tmp)*0.9)]
    result = []
    for i in range(len(cleaned_data)):
        a,b,c =  cleaned_data[i]
        if c >= level:
            # del cleaned_data[i]
            pass
        if c < level:
            result.append(cleaned_data[i])

    return result

