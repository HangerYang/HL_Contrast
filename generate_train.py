import numpy as np

with open('./generate_train.sh','w') as file:
    file.write("#!/bin/bash\n")
    for dataset in ["squirrel", "cora"]:
        for a in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            b = 1-a
            file.write("python train.py --config ./configs/fbgcn.json " 
                        "--dataset {} --preepoch 0 --a {} --b {} \n" .format(dataset, a, b) )