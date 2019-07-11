#!/usr/bin/env bash
python main.py --datasource=multidataset --metatrain_iterations=50000 --meta_batch_size=4 --update_batch_size=5 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=../Check_point/multidataset_5shot/ --num_filters=32 --max_pool=True --hidden_dim=128 --emb_loss_weight=0.01 --fix_embedding_sample=10

python main.py --datasource=multidataset --metatrain_iterations=50000 --meta_batch_size=4 --update_batch_size=5 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=../Check_point/multidataset_5shot/ --num_filters=32 --max_pool=True --hidden_dim=128 --emb_loss_weight=0.01 --test_set=True --test_epoch=49000 --train=False --test_dataset=1
