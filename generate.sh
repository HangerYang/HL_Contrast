#!/bin/bash
python MIX_FBGCN.py --config ./configs/fbgcn.json --dataset cora --pre_learning_rate 0.0005 --loss_type True --preepochs 300  --hidden_dim 256 
python MIX_FBGCN.py --config ./configs/fbgcn.json --dataset citeseer --pre_learning_rate 0.0005 --loss_type True --preepochs 300  --hidden_dim 256 
python MIX_FBGCN.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 300  --hidden_dim 256 
python MIX_FBGCN.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 300  --hidden_dim 256 
python MIX_FBGCN.py --config ./configs/fbgcn.json --dataset cora --pre_learning_rate 0.0005 --loss_type False --preepochs 300  --hidden_dim 256 
python MIX_FBGCN.py --config ./configs/fbgcn.json --dataset citeseer --pre_learning_rate 0.0005 --loss_type False --preepochs 300  --hidden_dim 256 
python MIX_FBGCN.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type False --preepochs 300  --hidden_dim 256 
python MIX_FBGCN.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type False --preepochs 300  --hidden_dim 256 
