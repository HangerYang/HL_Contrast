import numpy as np
with open('./generate_train.sh','w') as file:
    file.write("#!/bin/bash\n")
    for preepochs in range(50, 200, 50):
        for hidden_dim in [128, 256]:
            for aug in np.arange(0.2, 1., 0.1):
                for loss_type in ["True", "False"]:
                    for dataset in ["squirrel", "cora", "chameleon", "citeseer", "texas", ]:
                        file.write("python train_new.py --config ./configs/fbgcn.json " 
                                    "--dataset {} --preepochs {} "
                                    "--hidden_dim {} --loss_type {} "
                                    "--aug {} \n".format(dataset, preepochs, hidden_dim, loss_type, aug) )