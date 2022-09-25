#!/bin/bash
python GRACE.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.003 --preepochs 160  --hidden_dim 256

python GRACE.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0015 --preepochs 320  --hidden_dim 256

python GRACE.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0035 --preepochs 320  --hidden_dim 256

python GRACE.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.004 --preepochs 440  --hidden_dim 256

python GRACE.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0035 --preepochs 370  --hidden_dim 256

python GRACE.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.004 --preepochs 370  --hidden_dim 256

python GRACE.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0045 --preepochs 370  --hidden_dim 256

python GRACE.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0025 --preepochs 310  --hidden_dim 256

python GRACE.py --config ./configs/fbgcn.json --dataset texas --pre_learning_rate 0.0045 --preepochs 170  --hidden_dim 256

python GRACE.py --config ./configs/fbgcn.json --dataset texas --pre_learning_rate 0.0005 --preepochs 140  --hidden_dim 256

python GRACE.py --config ./configs/fbgcn.json --dataset texas --pre_learning_rate 0.001 --preepochs 140  --hidden_dim 256

python GRACE.py --config ./configs/fbgcn.json --dataset texas --pre_learning_rate 0.004 --preepochs 140  --hidden_dim 256


python BGRL.py --config ./configs/fbgcn.json --dataset cora --pre_learning_rate 0.0035 --preepochs 310  --hidden_dim 256
python BGRL.py --config ./configs/fbgcn.json --dataset cora --pre_learning_rate 0.0045 --preepochs 310  --hidden_dim 256
python BGRL.py --config ./configs/fbgcn.json --dataset cora --pre_learning_rate 0.002 --preepochs 400  --hidden_dim 256
python BGRL.py --config ./configs/fbgcn.json --dataset cora --pre_learning_rate 0.0015 --preepochs 310  --hidden_dim 256

python BGRL.py --config ./configs/fbgcn.json --dataset citeseer --pre_learning_rate 0.002 --preepochs 270  --hidden_dim 256
python BGRL.py --config ./configs/fbgcn.json --dataset citeseer --pre_learning_rate 0.002 --preepochs 290 --hidden_dim 256
python BGRL.py --config ./configs/fbgcn.json --dataset citeseer --pre_learning_rate 0.0035 --preepochs 130  --hidden_dim 256
python BGRL.py --config ./configs/fbgcn.json --dataset citeseer --pre_learning_rate 0.002 --preepochs 200  --hidden_dim 256

python BGRL.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.003 --preepochs 310  --hidden_dim 256
python BGRL.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0035 --preepochs 460 --hidden_dim 256
python BGRL.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0015 --preepochs 200  --hidden_dim 256
python BGRL.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0045 --preepochs 240  --hidden_dim 256

python BGRL.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.001 --preepochs 460  --hidden_dim 256
python BGRL.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0005 --preepochs 380 --hidden_dim 256
python BGRL.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.001 --preepochs 200  --hidden_dim 256
python BGRL.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.004 --preepochs 270  --hidden_dim 128

python BGRL.py --config ./configs/fbgcn.json --dataset texas --pre_learning_rate 0.0005 --preepochs 170 --hidden_dim 128
python BGRL.py --config ./configs/fbgcn.json --dataset texas --pre_learning_rate 0.0005 --preepochs 230 --hidden_dim 256
python BGRL.py --config ./configs/fbgcn.json --dataset texas --pre_learning_rate 0.0015 --preepochs 280  --hidden_dim 128
python BGRL.py --config ./configs/fbgcn.json --dataset texas --pre_learning_rate 0.001 --preepochs 280 --hidden_dim 128

python DGI.py --config ./configs/fbgcn.json --dataset texas --pre_learning_rate 0.004 --preepochs 130 --hidden_dim 128
python DGI.py --config ./configs/fbgcn.json --dataset texas --pre_learning_rate 0.004 --preepochs 500 --hidden_dim 256
python DGI.py --config ./configs/fbgcn.json --dataset texas --pre_learning_rate 0.002 --preepochs 130  --hidden_dim 128
python DGI.py --config ./configs/fbgcn.json --dataset texas --pre_learning_rate 0.001 --preepochs 130 --hidden_dim 128

python DGI.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0035 --preepochs 460 --hidden_dim 128
python DGI.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.002 --preepochs 310 --hidden_dim 128
python DGI.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0035 --preepochs 400  --hidden_dim 256
python DGI.py --config ./configs/fbgcn.json --dataset squirrel --pre_learning_rate 0.0045 --preepochs 450 --hidden_dim 128

python DGI.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.004 --preepochs 400 --hidden_dim 256
python DGI.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0045 --preepochs 400 --hidden_dim 128
python DGI.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.0025 --preepochs 260  --hidden_dim 256
python DGI.py --config ./configs/fbgcn.json --dataset chameleon --pre_learning_rate 0.003 --preepochs 400 --hidden_dim 256