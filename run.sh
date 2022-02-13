#!/bin/bash

# python train.py --config ./configs/gcn.json
# python train.py --config ./configs/gcn.json --dataset citeseer
# python train.py --config ./configs/gcn.json --dataset pubmed
# python train.py --config ./configs/gcn.json --dataset cornell
# python train.py --config ./configs/gcn.json --dataset texas
# python train.py --config ./configs/gcn.json --dataset wisconsin
# python train.py --config ./configs/gcn.json --dataset chameleon
# python train.py --config ./configs/gcn.json --dataset squirrel
# python train.py --config ./configs/gcn.json --dataset actor


python train.py --config ./configs/fbgcn.json --dataset cornell
python train.py --config ./configs/fbgcn.json --dataset cornell --loss_type True

python train.py --config ./configs/fbgcn.json --dataset texas
python train.py --config ./configs/fbgcn.json --dataset texas --loss_type True

python train.py --config ./configs/fbgcn.json --dataset actor
python train.py --config ./configs/fbgcn.json --dataset actor --loss_type True

# python train.py --config ./configs/fbgcn.json --dataset wisconsin
# python train.py --config ./configs/fbgcn.json --dataset wisconsin --loss_type True

# python train.py --config ./configs/fbgcn.json
# python train.py --config ./configs/fbgcn.json --loss_type True

# python train.py --config ./configs/fbgcn.json --dataset chameleon
# python train.py --config ./configs/fbgcn.json --dataset chameleon --loss_type True

# python train.py --config ./configs/fbgcn.json --dataset squirrel
# python train.py --config ./configs/fbgcn.json --dataset squirrel --loss_type True

# python train.py --config ./configs/fbgcn.json --dataset citeseer
# python train.py --config ./configs/fbgcn.json --dataset citeseer --loss_type True

# python train.py --config ./configs/fbgcn.json --dataset pubmed
# python train.py --config ./configs/fbgcn.json --dataset pubmed --loss_type True