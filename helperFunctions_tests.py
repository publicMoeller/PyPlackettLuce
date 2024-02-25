import unittest
import helperFunctions
import PlackettLuce
import math


class validChoicesTests(unittest.TestCase):

    # Lists of valid choices
    def test_validChoices_simple(self):
        result = helperFunctions.validChoices([1,2,3])
        self.assertEqual(result,[[1],[2],[3],[1,2],[1,3],[2,3],[1,2,3]])

    def test_validChoices_empty(self):
        result = helperFunctions.validChoices([])
        self.assertEqual(result,[])
    
    def test_validChoices_oneElement(self):
        result = helperFunctions.validChoices([1])
        self.assertEqual(result,[[1]])

    def test_validChoices_TwoElements(self):
        result = helperFunctions.validChoices([1,11])
        self.assertEqual(result,[[1],[11],[1,11]])


class stepStrengthTests(unittest.TestCase):
    # Single step strengths
    def test_stepStrength_SimnpleStrength0(self):
        self.assertEquals(1 ,helperFunctions.stepStrength([0], [1,2],[1,2]))

    def test_stepStrength_SimpleStrength1(self):
        self.assertEquals(2,helperFunctions.stepStrength([1], [1,2],[1,2]))

    def test_stepStrength_SimpleStrength01(self):
        self.assertEquals(2*math.sqrt(2),helperFunctions.stepStrength([0,1], [1,2],[1,2]))



class stepProbabilityTests(unittest.TestCase):

    # Single step probabilities
    def test_stepProbability_CatchBadLength(self):
        self.assertRaises(ValueError, helperFunctions.stepProbability, [1],[1,2],[1])
    
    def test_stepProbability_SimnpleProbs0(self):
        self.assertEquals(1/(3+2*math.sqrt(2)),helperFunctions.stepProbability([0], [1,2],[1,2]))

    def test_stepProbability_SimnpleProbs1(self):
        self.assertEquals(2/(3+2*math.sqrt(2)),helperFunctions.stepProbability([1], [1,2],[1,2]))

    def test_stepProbability_SimnpleProbs01(self):
        self.assertEquals((2*math.sqrt(2))/(3+2*math.sqrt(2)),helperFunctions.stepProbability([0,1], [1,2],[1,2]))    


if __name__ == '__main__':
    unittest.main()