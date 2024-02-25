

from PlackettLuce import PlackettLuce
"""
Tiny demonstration file to show the following things:
- Fitting of a model in a simple case using the full functionality of ties and printing all the report types
- Relative probabilites of the first ranks are close to the observations
- Due to the nature of the model the observed rankings only account for a part of the probability mass
    - in this case about 56%
    - this is due to the model assumptions, all possible rankings account for 100% of the probability mass (Delts2 is large while most 2-ties are not reasonable votes for humans)
    - the granularity of the model ending at 'how probable are n-ties?' allows for probability mass to 'leak' into unintended regions of the PL distribution
"""

P = PlackettLuce('VermontHouse.toc')
P.report(reportType= 'coefficients')
P.report(reportType= 'rankings')
P.report(reportType= 'choices')



# Print probabilities for observed rankings
print('{2,3},{1,4}:',P.predict([2,1,1,2]))
print('{1,2},{3,4}:',P.predict([1,1,2,2]))
print('{3,4},{1,2}:',P.predict([2,2,1,1]))
print('{1,4},{2,3}:',P.predict([1,2,2,1]))
print('{2,4},{1,3}:',P.predict([2,1,2,1]))
print('{1,3},{2,4}:',P.predict([1,2,1,2]))

print('3,{1,2,4}:',P.predict([2,2,1,2]))
print('2,{1,3,4}:',P.predict([2,1,2,2]))
print('4,{1,2,3}:',P.predict([2,2,2,1]))
print('1,{2,3,4}:',P.predict([1,2,2,2]))

print('1,2,3,4:',P.predict([1,2,3,4]))


# Sum of probabilites for the observed rankings
probabilityMass = 0
rankings =[[2,1,1,2],[1,1,2,2],[2,2,1,1],[1,2,2,1],[2,1,2,1],[1,2,1,2],[2,2,1,2],[2,1,2,2],[1,2,2,2],[2,2,2,1],[1,2,3,4]]
for i in rankings:
    probabilityMass += P.predict(i)
print('Probabiliy mass of observed rankings',probabilityMass)

# All 75 Rankings for 4 Options with arbitrary ties
#(FindRepalce is just faster for one time use)
TotalRankings = [[1,1,1,1], # total tie    
                                   
                 [2,1,1,2], # 6 Rankings of 2,2 pratitions
                 [1,1,2,2],
                 [2,2,1,1],
                 [1,2,2,1],
                 [2,1,2,1],
                 [1,2,1,2],

                 [2,2,1,2], # 4 times 1,3 partition
                 [2,1,2,2],
                 [1,2,2,2],
                 [2,2,2,1],

                 [1,1,1,2], # 4 times 1,3 partition 
                 [1,1,2,1],
                 [1,2,1,1],
                 [2,1,1,1],

                 [1,1,2,3],# 12 times 2,1,1 Partition
                 [1,1,3,2],
                 [1,2,1,3],
                 [1,3,1,2],
                 [1,2,3,1],
                 [1,3,2,1],
                 [2,1,1,3],
                 [3,1,1,2],
                 [2,1,3,1],
                 [3,1,2,1],
                 [2,3,1,1],
                 [3,2,1,1],

                 [2,2,1,3], # 12 times 1,2,1 Partition
                 [2,2,3,1],
                 [2,1,2,3],
                 [2,3,2,1],
                 [2,1,3,2],
                 [2,3,1,2],
                 [1,2,2,3],
                 [3,2,2,1],
                 [1,2,3,2],
                 [3,2,1,2],
                 [1,3,2,2],
                 [3,1,2,2],
                 
                 [3,3,1,2],# 12 times 1,1,2 Partition
                 [3,3,2,1],
                 [3,1,3,2],
                 [3,2,3,1],
                 [3,1,2,3],
                 [3,2,1,3],
                 [1,3,3,2],
                 [2,3,3,1],
                 [1,3,2,3],
                 [2,3,1,3],
                 [1,2,3,3],
                 [2,1,3,3],

                 [1,2,3,4],# 24 times 1,1,1,1 Partition
                 [1,2,4,3],
                 [1,3,2,4],
                 [1,3,4,2],
                 [1,4,2,3],
                 [1,4,3,2],
                 [2,1,3,4],
                 [2,1,4,3],
                 [2,3,1,4],
                 [2,3,4,1],
                 [2,4,1,3],
                 [2,4,3,1],
                 [3,1,2,4],
                 [3,1,4,2],
                 [3,2,1,4],
                 [3,2,4,1],
                 [3,4,1,2],
                 [3,4,2,1],
                 [4,1,2,3],
                 [4,1,3,2],
                 [4,2,1,3],
                 [4,2,3,1],
                 [4,3,1,2],
                 [4,3,2,1]]

TotalProbabilityMass = 0
for i in TotalRankings:
    TotalProbabilityMass += P.predict(i)
print('Total probability mass',TotalProbabilityMass)



