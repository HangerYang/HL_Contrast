<<<<<<< HEAD
import numpy as np

with open('./train_texas.sh','w') as file:
    file.write("#!/bin/bash\n")
    for dataset in ["texas"]:
        for loss_type in ["True", "False"]:
            for preepochs in range(0, 350, 50):
                for hidden_dim in [128, 256]:
                    file.write("python train.py --config ./configs/fbgcn.json " 
                    "--dataset {} --pre_learning_rate {} " 
                    "--loss_type {} --preepochs {}  --hidden_dim {} \n".format(dataset, 0.0005,loss_type, preepochs, hidden_dim))
=======
import numpy as np

with open('./generate.sh','w') as file:
    file.write("#!/bin/bash\n")
    for preepochs in [200, 300]:
        for loss_type in ["True", "False"]:
            for dataset in ["texas"]:
                for aug_type in ["FD", "FM"]:
                    for aug_side in ["both", "high", "low"]:
                        for aug in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                            file.write("python MIX_FBGCN.py --config ./configs/fbgcn.json " 
                            "--dataset {} --pre_learning_rate {} " 
                            "--loss_type {} --preepochs {}  --hidden_dim {} "
                            "--aug_type {}  --aug_side {} --aug {} \n".format(dataset, 0.0005, loss_type, preepochs, 256, aug_type, aug_side, aug) )
>>>>>>> 809d5155a8b1bdc4f5814cd60aafe9df06760dd5

with open('./train_texas.sh','w') as file:
    file.write("#!/bin/bash\n")
    for dataset in ["texas"]:
        for loss_type in ["True", "False"]:
            for preepochs in range(0, 350, 50):
                for hidden_dim in [128, 256]:
                    file.write("python train.py --config ./configs/fbgcn.json " 
                    "--dataset {} --pre_learning_rate {} " 
                    "--loss_type {} --preepochs {}  --hidden_dim {} \n".format(dataset, 0.0005,loss_type, preepochs, hidden_dim))