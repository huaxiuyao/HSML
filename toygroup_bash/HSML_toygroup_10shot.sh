#!/usr/bin/env bash
python main.py --datasource=mixture --metatrain_iterations=70000 --norm=None --update_batch_size=10 --update_batch_size_eval=10 --resume=False --num_updates=5 --logdir=../Check_point/syncgroup_10shot --emb_loss_weight=0.01 --hidden_dim=40

python main.py --datasource=mixture --metatrain_iterations=70000 --norm=None --update_batch_size=10 --update_batch_size_eval=10 --resume=False --num_updates=5 --logdir=../Check_point/syncgroup_10shot --emb_loss_weight=0.01 --hidden_dim=40 --test_set=True --test_epoch=68000 --train=False --num_test_task=4000