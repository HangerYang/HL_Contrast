import numpy as np

with open('./generate.sh','w') as file:
    file.write("#!/bin/bash\n")
    for preepochs in range(20, 300, 10):
        for loss_type in ["True", "False"]:
            for dataset in ["citeseer"]:
                for hidden_dim in [32, 64, 128, 256]:
                    for pre_learning_rate in np.arange(0.0005, 0.005, 0.0005):
                        file.write("python FBGCN.py --config ./configs/fbgcn.json " 
                        "--dataset {} --pre_learning_rate {} " 
                        "--loss_type {} --preepochs {}  --hidden_dim {} \n".format(dataset, pre_learning_rate, loss_type, preepochs, hidden_dim) )