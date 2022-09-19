#!/bin/bash
python MIX_FBGCN_AUG.py --config ./configs/fbgcn.json --dataset citeseer --pre_learning_rate 0.0005 --loss_type True --preepochs 200  --hidden_dim 256 --second_hidden_dim 128 --aug_type ER --aug_side both --aug 0.5
python MIX_FBGCN_AUG_SEQ.py --config ./configs/fbgcn.json --dataset citeseer --pre_learning_rate 0.0005 --loss_type True --preepochs 200  --hidden_dim 256 --second_hidden_dim 256 --aug_type ER --aug_side both --aug 0.5
python MIX_FBGCN_AUG_SEQ.py --config ./configs/fbgcn.json --dataset citeseer --pre_learning_rate 0.0005 --loss_type True --preepochs 200  --hidden_dim 128 --second_hidden_dim 128 --aug_type ER --aug_side both --aug 0.5

python MIX_FBGCN_AUG.py --config ./configs/fbgcn.json --dataset cora --pre_learning_rate 0.0005 --loss_type False --preepochs 200  --hidden_dim 256 --second_hidden_dim 128 --aug_type ER --aug_side both --aug 0.8
python MIX_FBGCN_AUG_SEQ.py --config ./configs/fbgcn.json --dataset cora --pre_learning_rate 0.0005 --loss_type False --preepochs 200  --hidden_dim 256 --second_hidden_dim 256 --aug_type ER --aug_side both --aug 0.8  
python MIX_FBGCN_AUG_SEQ.py --config ./configs/fbgcn.json --dataset cora --pre_learning_rate 0.0005 --loss_type False --preepochs 200  --hidden_dim 128 --second_hidden_dim 128 --aug_type ER --aug_side both --aug 0.8 

python MIX_FBGCN.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 200  --hidden_dim 256 --second_hidden_dim 128 --aug_type FD --aug_side both --aug 0.2 
python MIX_FBGCN_SEQ.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 200  --hidden_dim 256 --second_hidden_dim 256 --aug_type FD --aug_side both --aug 0.2 
python MIX_FBGCN_SEQ.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0005 --loss_type True --preepochs 200  --hidden_dim 128 --second_hidden_dim 128 --aug_type FD --aug_side both --aug 0.2 

python MIX_FBGCN.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 200  --hidden_dim 256 --second_hidden_dim 128 --aug_type FD --aug_side both --aug 0.8
python MIX_FBGCN_SEQ.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 200  --hidden_dim 256 --second_hidden_dim 256 --aug_type FD --aug_side both --aug 0.8
python MIX_FBGCN_SEQ.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --loss_type True --preepochs 200  --hidden_dim 128 --second_hidden_dim 128 --aug_type FD --aug_side both --aug 0.8     

python MIX_FBGCN.py --config ./configs/fbgcn.json --dataset texas --pre_learning_rate 0.0005 --loss_type False --preepochs 100  --hidden_dim 128 --second_hidden_dim 64 --aug_type FD --aug_side both --aug 0.4 
python MIX_FBGCN_SEQ.py --config ./configs/fbgcn.json --dataset texas --pre_learning_rate 0.0005 --loss_type False --preepochs 100  --hidden_dim 128 --second_hidden_dim 128 --aug_type FD --aug_side both --aug 0.4 
python MIX_FBGCN_SEQ.py --config ./configs/fbgcn.json --dataset texas --pre_learning_rate 0.0005 --loss_type False --preepochs 100  --hidden_dim 64 --second_hidden_dim 64 --aug_type FD --aug_side both --aug 0.4 