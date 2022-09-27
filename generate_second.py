import numpy as np

with open('./squirrel_train_aug.sh','w') as file:
    file.write("#!/bin/bash\n")
    for preepochs in range(0,350, 50):
        for loss_type in ["True", "False"]:
            for dataset in ["squirrel"]:
                for aug_type in ["FM"]:
                    for hidden_dim in [128, 256]:
                        for aug_side in ["low"]:
                            for aug in [0.1, 0.2, 0.3]:
                                    file.write("python train_shared_weights.py --config ./configs/fbgcn.json " 
                                    "--dataset {} --pre_learning_rate {} " 
                                    "--loss_type {} --preepochs {}  --hidden_dim {} --second_hidden_dim {} "
                                    "--aug_type {}  --aug_side {} --aug {} \n".format(dataset, 0.0005, loss_type, preepochs, hidden_dim, hidden_dim, aug_type, aug_side, aug))
