"""
This module provides functionality for testing the accuracy of the
eng-->fra neural machine translation model defined in translator_core
"""

from random import random
from translator_inference import Model

# Promt user for evaluation parameters
CASES = int(input("Number of evaluation cases:  "))

# Evaluate on cases
print("Loading model...")
model = Model()

for i in range(CASES):
    index = int(random() * len(model.pairs))
    print("English:     {}".format(model.pairs[index][0]))
    print("French:      {}".format(model.pairs[index][1]))
    print("Translation: {}".format(model.translate(model.pairs[index][0])))
    print()
