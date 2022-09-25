import numpy as np

with open('./generate_GRACE.sh','w') as file:
    file.write("#!/bin/bash\n")
    for preepochs in range(100, 410, 10):
            for dataset in ["texas", "chameleon", "squirrel"]:
                for hidden_dim in [128, 256]:
                    for pre_learning_rate in np.arange(0.0005, 0.005, 0.0005):
                        file.write("python GRACE.py --config ./configs/fbgcn.json " 
                        "--dataset {} --pre_learning_rate {} " 
                        "--preepochs {}  --hidden_dim {} \n".format(dataset, pre_learning_rate, preepochs, hidden_dim))