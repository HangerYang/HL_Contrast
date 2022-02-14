import numpy as np

with open('./generate.sh','w') as file:
    file.write("#!/bin/bash\n")
    for preepochs in range(0, 150, 10):
        for loss_type in ["True", "False"]:
            for hidden_dim in [32, 64, 128, 256]:
                for learning_rate in np.arange(0.0005, 0.01, 0.0005):
                    for pre_learning_rate in np.arange(0.0005, 0.001, 0.0005):
                        file.write("python train.py --config ./configs/fbgcn.json" 
                        "--dataset cora --learning_rate {} --pre_learning_rate {}" 
                        "--hidden_dim {} --loss_type {} --preepochs {} \n".format(learning_rate, pre_learning_rate, hidden_dim, loss_type, preepochs) )