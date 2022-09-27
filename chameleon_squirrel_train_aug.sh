#!/bin/bash
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 0  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 0  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 0  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.4 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 0  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 0  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 0  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.4 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 0  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 0  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 0  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.4 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 0  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 0  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 0  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.4 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 50  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 50  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 50  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.4 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 50  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 50  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 50  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.4 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 50  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 50  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 50  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.4 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 50  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 50  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 50  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.4 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 100  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 100  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 100  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.4 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 100  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 100  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 100  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.4 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 100  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 100  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 100  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.4 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 100  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 100  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 100  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.4 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 150  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 150  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 150  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.4 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 150  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 150  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 150  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.4 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 150  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 150  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 150  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.4 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 150  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 150  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 150  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.4 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 200  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 200  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 200  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.4 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 200  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 200  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 200  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.4 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 200  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 200  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 200  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.4 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 200  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 200  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 200  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.4 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 250  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 250  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 250  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.4 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 250  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 250  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 250  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.4 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 250  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 250  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 250  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.4 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 250  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 250  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 250  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.4 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 300  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 300  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 300  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.4 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 300  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 300  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 300  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.4 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 300  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 300  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 300  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.4 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 300  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 300  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 300  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.4 

python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 0  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.1 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 0  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 0  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 0  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.1 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 0  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 0  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 0  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.1 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 0  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 0  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 0  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.1 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 0  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 0  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 50  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.1 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 50  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 50  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 50  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.1 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 50  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 50  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 50  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.1 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 50  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 50  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 50  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.1 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 50  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 50  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 100  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.1 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 100  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 100  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 100  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.1 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 100  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 100  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 100  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.1 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 100  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 100  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 100  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.1 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 100  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 100  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 150  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.1 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 150  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 150  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 150  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.1 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 150  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 150  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 150  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.1 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 150  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 150  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 150  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.1 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 150  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 150  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 200  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.1 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 200  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 200  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 200  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.1 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 200  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 200  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 200  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.1 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 200  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 200  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 200  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.1 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 200  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 200  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 250  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.1 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 250  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 250  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 250  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.1 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 250  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 250  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 250  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.1 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 250  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 250  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 250  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.1 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 250  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 250  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 300  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.1 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 300  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 300  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 300  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.1 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 300  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 300  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 300  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.1 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 300  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 300  --hidden_dim 128 --second_hidden_dim 128 --aug_type FM  --aug_side low --aug 0.3 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 300  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.1 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 300  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.2 
python train_shared_weights.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 300  --hidden_dim 256 --second_hidden_dim 256 --aug_type FM  --aug_side low --aug 0.3 
