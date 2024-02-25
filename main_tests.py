import unittest
import pandas as pd
import PlackettLuce


class pruneDeltasTests(unittest.TestCase):

    def test_doNothing(self):
        # Nothing should be pruned
        P = PlackettLuce.PlackettLuce('Netflix.soc')
        testee = pd.DataFrame([[1,1,1],[1,1,1],[1,1,1]], columns = ['A','delta674465','outcome'])
        pruned = P.pruneDeltas(testee)
        self.assertTrue(testee.equals(pruned))

    def test_simplePruning(self):
        # middle column should be deleted
        P = PlackettLuce.PlackettLuce('Netflix.soc')
        testee = pd.DataFrame([[1,0,1],[1,0,1],[1,0,1]], columns = ['A','delta674465','outcome'])
        trueResult = pd.DataFrame([[1,1],[1,1],[1,1]], columns = ['A','outcome'])
        pruned = P.pruneDeltas(testee)
        self.assertTrue(pruned.equals(trueResult))
        

    def test_ComplexCase(self):
         # close to real world example, delta 1 and 2 should drop, checks that nonempty can indeed be correctly dropped
        P = PlackettLuce.PlackettLuce('Netflix.soc')
        testeeMatrix = [[1,2,3,0,5,0],
                        [10,8,0,0,11,1],
                        [10,1,0,0,17,18]]
        trueResultMatrix= [[1,2,5,0],
                          [10,8,11,1],
                          [10,1,17,18]]
        testee = pd.DataFrame(testeeMatrix, columns = ['A','B','delta1','delta2','delta3','outcome'])
        trueResult = pd.DataFrame(trueResultMatrix, columns = ['A','B','delta3','outcome'])
        pruned = P.pruneDeltas(testee)
        self.assertTrue(pruned.equals(trueResult))

if __name__ == '__main__':
    unittest.main()