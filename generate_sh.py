with open('./train_texas.sh','w') as file:
    file.write("#!/bin/bash\n")
    for dataset in ["texas"]:
        for loss_type in ["True", "False"]:
            for preepochs in range(0, 350, 50):
                for hidden_dim in [128, 256]:
                    file.write("python train.py --config ./configs/fbgcn.json " 
                    "--dataset {} --pre_learning_rate {} " 
                    "--loss_type {} --preepochs {}  --hidden_dim {} \n".format(dataset, 0.0005,loss_type, preepochs, hidden_dim))