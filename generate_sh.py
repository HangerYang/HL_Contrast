import numpy as np

with open('./generate.sh','w') as file:
    file.write("#!/bin/bash\n")
    for preepochs in [300]:
        for loss_type in ["True", "False"]:
            for dataset in ["cora", "citeseer", "squirrel", "chameleon"]:
                for hidden_dim in [256]:
                    for pre_learning_rate in [0.0005]:
                        file.write("python MIX_FBGCN.py --config ./configs/fbgcn.json " 
                        "--dataset {} --pre_learning_rate {} " 
                        "--loss_type {} --preepochs {}  --hidden_dim {} \n".format(dataset, pre_learning_rate, loss_type, preepochs, hidden_dim) )