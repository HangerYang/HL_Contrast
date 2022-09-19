import numpy as np

with open('./generate_second.sh','w') as file:
    file.write("#!/bin/bash\n")
    for preepochs in [200, 300]:
        for loss_type in ["True", "False"]:
            for dataset in ["texas"]:
                for aug_type in ["ER", "ND"]:
                    for aug_side in ["both", "high", "low"]:
                        for aug in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                            file.write("python MIX_FBGCN_AUG.py --config ./configs/fbgcn.json " 
                            "--dataset {} --pre_learning_rate {} " 
                            "--loss_type {} --preepochs {}  --hidden_dim {} "
                            "--aug_type {}  --aug_side {} --aug {} \n".format(dataset, 0.0005, loss_type, preepochs, 256, aug_type, aug_side, aug) )