
import numpy as np
import pandas as pd

import re
import sys
from helperFunctions import stepProbability,stepStrength

# Binomial coeficcient
from math import exp

# For connectivity checking
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components




# Allow printing of arbitrary size for np
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=2)


# Statsmodels 

from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import (Exchangeable,Independence,Autoregressive)
from statsmodels.genmod.families import Poisson
from statsmodels.genmod.cov_struct import (Exchangeable,Independence,Autoregressive, GlobalOddsRatio)
import statsmodels.api as sm



class PlackettLuce:
    """
    Fits a PlackettLuce model with tie support using the DavidsonLuce extension.
    Inspired by the R package https://hturner.github.io/PlackettLuce/

    All four preflib ORDINAL file types( .xOx) be parsed correctly, since we are not assuming complete lists and strict order syntax is a subset of the syntax used for ties.


    Attributes:
        filepath(string): filepath to record. 
        rankings(np.array): Input orderings rewritten as rankings
        weights(list): Number of occourences for each ranking
        choices: Stores the exoanded form for fitting
        names(list of strings): Alternative names of the ranked Items 
        alternativesCount(int): Number of ranked alternatives
        rankingsCount(int): Number of input rankings
    



    """

    def __init__(self, filepath:str, nPseudo = 0.5) -> None:
        """
        Reads Preflib .toc/.toi/.soc/.soi file in and fits a PlackettLuce model

        nPseudo: Strength of used pseudoitems. 0 to turn off. Necessary for disconnected networks.

        See citations at https://hturner.github.io/PlackettLuce/ for more information about the model
        """
        
        # path to record 
        self.filepath = filepath
        # save weights for our pseudoItem
        self.nPseudo = nPseudo
        # Input rankings np.array
        self.rankings = None
        # number of votes per Ranking
        self.weights = []
        # Choices calculated from preference data
        self.choices = None
        # Number of Alternativesd
        self.alternativesCount = None
        # Number of unique orders = lines of ranking data
        self.rankingsCount = None
        # dict to resolve alternative names
        self.names = []
        # Fitted model
        self.model = None

        self.columnNames = None
        
  


        # read in the indicated file
        self._read()

        # set up adjacency
        self.adjacencyMatrix = np.zeros((self.alternativesCount,self.alternativesCount), dtype=bool)

        self.fit()

    
    
    def _read(self) -> None:
        """
        Reads in the ordinal preferences file types from Preflib see:
        https://www.preflib.org/format

        Stores all the needed metadata in appropriate structures in the PlackettLuce class and the ordinal preference data to rankings. 
        
        Example
            A ranking of four items is transformed from
            2, {1,3}
            to 
            2   1   2   0    
        """
        toImport = open(self.filepath, 'r')
    
        # compile needed patterns
        # for capturing and deleting the #ALTERNATIVE NAME.. part
        namePattern = re.compile('^\#\w* ALTERNATIVE NAME \d+\: ') 
        # for capturing the count in the acutal data
        leadingDigits = re.compile('^\d+')
        # for cutting away the count:
        dataPattern = re.compile('^\d+\:')
        # finding the number of Alternatives
        numberAlternatives = re.compile('^\#\w* NUMBER ALTERNATIVES\:\w*')
        # finding the number of unique Orders = lines of ranking data
        numberRankings = re.compile('^\#\w* NUMBER UNIQUE ORDERS\:\w*')
        # opening and closing brackets
        leadingTie = re.compile('^\{')
        leadingTieClose = re.compile('^\}')
        
        # flag to skip most matching once in the data semgent.
        inDataSegment = False 
        for line in toImport:
            # Find and handle altName lines
            if not inDataSegment and namePattern.match(line):
                altName = namePattern.sub('', line)
                altName = altName.rstrip()
                self.names.append(altName.replace(' ','_').replace(':','_').replace('-','_').replace('\'','_').replace('.','_'))

            # Find and handle number of Alternatives
            if not inDataSegment and numberAlternatives.match(line):
                count = numberAlternatives.sub('', line)
                self.alternativesCount = int(count)

            # Find and handle number unique orders
            if not inDataSegment and numberRankings.match(line):
                count = numberRankings.sub('', line).rstrip()
                self.rankingsCount = int(count)


            # Get input and weights from all lines with actual data
            if dataPattern.match(line):
                count = None
                count = int(leadingDigits.match(line).group())
                line = dataPattern.sub('',line)
                line = line.lstrip().rstrip()
                self.weights.append(count)
                # Initialize array if first line
                if self.rankings is None:
                    #Stop looking for more header lines
                    inDataSegment = True 
                    self.rankings = np.full((self.rankingsCount, self.alternativesCount), 0)
                    writePosition = 0
                # Write the actual ranking on the appropirate line
                nextRank = 1
                currentRanking = np.full(self.alternativesCount, 0)
                tie = False
                while line:
                    if leadingDigits.match(line):
                        rankee = int(leadingDigits.match(line).group())
                        line = leadingDigits.sub('',line).lstrip(',')
                        currentRanking[rankee-1] = nextRank
                        if not tie:
                            nextRank += 1
                    if leadingTie.match(line):
                        tie = True
                        line = line.lstrip('{')
                    if leadingTieClose.match(line):
                        tie = False
                        line = line.lstrip('}')
                        nextRank += 1
                    line = line.lstrip(',')
                    


                self.rankings[writePosition] = currentRanking
                writePosition += 1
        toImport.close()



    def connectivity(self) -> int:
        """
        Checks if implied DAG is disconnected and returns number of disconnected components
        """
        

        #We actually only set cells to 1 to signify 'is connected'. We are not interested in specific numbers here. Then we use scipy.sparse.csgraph.connected_components to check for number of components.
        for line in range(self.rankingsCount):
            for element in range(self.alternativesCount):
                for sample in range (self.alternativesCount):
                    if element == sample:
                        continue
                    if self.rankings[line][element] > self.rankings[line][sample]:
                        # add weighted record that element beats sample
                        # No +-1, both are 0 indexed!
                        self.adjacencyMatrix[element][sample] = True
                        # Was there a reason to calculate actual weights?
                        #+= self.weights[line]

        # uses CSR format
        # assumes directed = True by default.
        components = connected_components(self.adjacencyMatrix , directed=True, connection='strong', return_labels=False)
        return(components)



    def expand(self) -> None:
        """
        Expands Rankings to the representation needed for fitting the model. 

        Very minimal example in "Davidson-Luce model for multi-item choice with ties" appendix
        https://arxiv.org/pdf/1909.07123.pdf

        """
        

        # one line per possibly occouring tie (including 0th order). 
        linesPerRanking =  pow(2,self.alternativesCount)-1
        noLines =  self.rankingsCount * linesPerRanking
        noColumns = 2 * self.alternativesCount + 1

        
        self.choices = np.zeros((noLines,noColumns))
        

        # Initialize all the lines (count 0  is a relevant observation too, so every line has to be initialized)
        for linenumber in range(len(self.choices)):
            encodes = (linenumber % linesPerRanking) +1

    
            binary = (format(encodes,"0"+str(self.alternativesCount)+"b"))
            ones = binary.count('1')
            # Ranking number
            self.choices[linenumber][0] = (linenumber // linesPerRanking) + 1
            # tie variables
            if ones > 1:
                self.choices[linenumber][self.alternativesCount - 1 + ones] = 1

            # Probabilities
            for i in range(len(binary)):
                if int(binary[i]) != 0:
                    self.choices[linenumber][i+1] = int(binary[i])/ones
        
        # add the counts

        # so that we can just write each line to its place in the binary order + rankingOfset*2n-1
        rankingsOffset = 0
        for ranking in self.rankings:

            #print(ranking)
            countRanked = (np.count_nonzero(np.asarray(ranking) > 0)) #only nonzero does not account for nan
            #print(countRanked)
            maxRank = int((np.nanmax(ranking)))
            for rank in range(1, maxRank + 1):   
                    line = None
                    # no win, if last and every entry ranked (That was wrong, we would lose a lot of data that way)
                    #if countRanked == self.alternativesCount and maxRank == rank:
                    #    continue
                            
                    # find line for encoding
                    signature = (np.asarray(ranking) == rank)
                    line = -1
                    for i in range(len(signature)):
                        line += signature[-(i+1)]*pow(2,i)
                    # add the count
                    self.choices[rankingsOffset * linesPerRanking + line][-1] += self.weights[rankingsOffset]*(maxRank + 1 - rank)

            rankingsOffset += 1
        


    def _pseudoitem(self) -> None:
        """
        Adds pseudoitem with connectivity strength nPseudo to the rankings
        """
        #separate array to append only once (numpy copies for that)
        pseudoRankings = np.zeros((2*self.alternativesCount, self.alternativesCount + 1))
        #add pseudorankings
        for i in range(self.alternativesCount):
            # pseudo item loses
            pseudoRankings[i][i] = 1
            pseudoRankings[i][-1] = 2
            # pseudo item wins
            pseudoRankings[i + self.alternativesCount][i] = 2
            pseudoRankings[i + self.alternativesCount][-1] = 1

        
        # self.rankings is missing a column for the pseudoitem
        z = np.zeros((self.rankingsCount,1), dtype=float)
        self.rankings = np.append(self.rankings, z, axis=1)
        # Stack with pseudoitems
        self.rankings = np.vstack((self.rankings,pseudoRankings))
        # fill up weight vector
        extraWeights = np.full((1,2*self.alternativesCount), self.nPseudo)
        self.weights = np.append(self.weights,extraWeights)
        

        #change relevant statistics 
        self.rankingsCount += 2 * self.alternativesCount
        self.alternativesCount += 1
        self.names.append('pseudoitem')
        


    def fit(self) -> None:
        """
        Does the actucal fitting
        """
        
        if self.nPseudo:
            self._pseudoitem()
        else:
            if self.connectivity() > 1:
                raise ValueError('disconected network try not setting npseudo to 0')
       
        self.expand()
        
        # Generate list of column names
        deltaNames = list()
        delta = 2
        while len(deltaNames) + len(self.names ) + 2 < len(self.choices[0]):
            deltaNames.append("delta_" + str(delta))
            delta += 1 
        columNames = ['comparison'] + self.names + deltaNames + ['outcome']

        

        
        data = pd.DataFrame(self.choices, columns = columNames)
        # Drop Deltas that do not occour
        data = self.pruneDeltas(data)   
        #data = data.drop(['pseudoitem'], axis=1)
    

        # Generate Formula for fitting
        formula = 'outcome ~  comparison '
        for i in data.columns:
            if i == ('outcome'):
                continue
            if i == ('comparison'):
                continue
            else:
               formula = formula + ' + ' + i + ' '

        # fitting
        
        fam = Poisson()
        ind = Independence()
        
        model2 =  sm.formula.glm(formula,family=sm.families.Poisson(), data=data)#, cov_struct = ind)
        result2 = model2.fit(method = "lbfgs", maxiter=10000)
        self.model = result2
        


    def pruneDeltas(self, data:pd.DataFrame) -> pd.DataFrame:
        """
        Prunes the columns starting with delta if no outcomes containing them are observed
        """
        droplist = []
        found = False
        for column in data:
            if column.startswith('delta'):
                found = False
                for line in data.index:
                    if data.iloc[line].at['outcome'] and data.iloc[line].at[column]:
                        # A line where we witness a n-Tie occouring
                        found = True
                if found:
                    # do not check all lines of that column
                    continue
                if not found:
                    droplist.append(column)  
        # actually drop
        data = data.drop(droplist, axis=1)
        return data
    


    def predict(self, ranking:list) -> float:
        """
        predicts probability from fitted model and given ranking

        Predicts the Probability for a ranking on basis of fitted PlackettLuce Object.
        Unranked Items need to be denoted as 0 not as last place. 
        Example: Correct form for 5 items ranked 1,{2,4},3 would be [1,2,4,2,0]
        """
    


        # check if length matches items (-pseudoitem)
        if 'pseudoitem' in self.names:
            if len(ranking) != len(self.names)-1:
                raise ValueError('your ranking does not seem to have the right length')
        else:
            if len(ranking) != len(self.names):
                raise ValueError('your ranking does not seem to have the right length')
            
        # check type
        if not all(isinstance(val, int) for val in ranking):
            raise ValueError('only whole numbers please')
        # check if numbering is continuous
        check = ranking.copy()
        check.sort()
        if check[0] < 0 or check[0] >1:
            raise ValueError('continuous numbers please')
        
        last = check[0]
        for i in check:
            if i == last:
                continue
            elif i == last+1:
                last = i
                continue
            else:
                raise ValueError('continuous numbers please')
                
        # Are we fitted?
        if self.model == None:
            raise ValueError('please fit model first')
        

        # get number of occourences, places, highest.
        completeIndices = []
        zeroes = [num for num, x in enumerate(ranking) if x == 0]
        for i in range(1, last + 1):
            indices = [num for num, x in enumerate(ranking) if x == i]
            completeIndices.append(indices)
        # completeIndices is now a List with i lists for the indices of alternatives that are on ith rank
        # get normal coefs and deltas (with 0 for missing deltas)
        coefs = []
        deltas = [1] # delta1 = 1 allows to treat single ranks as ties of order 1
        for i in self.model.params.index:
            if i == 'pseudoitem' or i == 'Intercept' or i == "comparison":
                continue
            elif i.startswith('delta_'):
                while int(i.replace('delta_',''))-1 > len(deltas):
                    deltas.append(0) #fill intermediate deltas that do not occour
                deltas.append(exp(self.model.params.loc[i]))
                
            else:
                coefs.append(exp(self.model.params.loc[i]))
            # If deltas in the end were not availible we need to fill up
        while len(coefs) > len(deltas):
            deltas.append(0) #fill intermediate deltas that do not occour
            
        
        
        

    
        # set strength of unranked alternatives to 0
        for i in zeroes:
            coefs[i] = 0

        #calculate the actual probability
        probability = 1
        while completeIndices: # iterates over ranks

            probability *= stepProbability(completeIndices[0],coefs,deltas)

            
            # mark this rank as done
            rank = completeIndices.pop(0)
            # set strength of used alternatives to 0
            for i in rank:
                coefs[i] = 0
        return probability



    def report(self, reportType:str = 'coefficients') -> None:
        """
        Prints the corresponding report for coefficients','rankings' and 'choices'

        Defaults to 'coefficients' because this it probably used most.
        """

        # Are we fitted?
        if self.model == None:
                raise ValueError('please fit model first')
        match reportType:
            case 'coefficients':
                print(self.model.summary())
            case 'rankings':
                print(self.rankings)
            case 'choices':
                print(pd.DataFrame(self.choices))#.to_string())
            case other:
                print("specify the report you want. Options are:'coefficients','rankings','choices'")

        

if __name__ == '__main__':
    pass