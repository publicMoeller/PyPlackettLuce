import itertools

# helpful Math for calculating the strengths and probabilities from the coefficients

def validChoices(iterable) -> list:
    """
    Gives all NONEMPTY subsets. (itertools.combinations only creates constant length combinations)

    modified from itertools recipies powerset: https://docs.python.org/3/library/itertools.html#itertools-recipes
        Changed:
        - only nonempty
        - iterable casted to list
        - inner list instead of tuples
    
    """

    s = list(iterable)
    raw = list(itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1)))
    raw.pop(0)
    result = []
    for i in raw:
        result.append(list(i))

    return result
    
def stepStrength(choice:list, coefs:list, deltas:list) -> float:
    """
    Computes STRENGTH for choosing all the indices in choice given coeffs and deltas

    Assumes values are not log anymore.
    """
    if len(coefs) != len(deltas):
        raise ValueError('you should have as many deltas as coefficients')
    
    rawstrength = 1
    for i in choice:
        rawstrength *= coefs[i]
    strength = deltas[len(choice)-1]* pow(rawstrength, 1/len(choice))
    return strength



def stepProbability(choice:list, coefs:list, deltas:list) -> float:
    """
    Computes PROBABILITY for choosing all the indices in choice given coeffs and deltas

    Assumes values are not log anymore.
    """
    choices = validChoices(range(len(coefs)))
    strengths = []
    for i in range(len(choices)):
        strengths.append(stepStrength(choices[i],coefs,deltas))
    normFactor = sum(strengths)
    
    probability = strengths[choices.index(choice)] / normFactor

    return probability