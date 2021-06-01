import random
import sys
import os

path_to_raw_complexes = sys.argv[1]
test_size = float(sys.argv[2])

complexes_names = [dirname for dirname in os.listdir(path_to_raw_complexes)]

test_complexes_names = random.sample(complexes_names, int(test_size * len(complexes_names)))

with open('test_complexes.txt', 'w') as f:
    for line in test_complexes_names:
        f.write(line + '\n')
