#!/bin/sh
# PURPOSE: Runs predict.py with all inputs to test for bugs
#
# Usage: sh run_models_batch.sh    -- will run program from commandline

python predict.py flower/test/1/image_06743.jpg checkpoints/dn_default.pth

python predict.py flower/test/1/image_06743.jpg checkpoints/an_default.pth --arch alexnet

python predict.py flower/test/1/image_06743.jpg checkpoints/an_default.pth --arch alexnet

python predict.py flower/test/1/image_06743.jpg checkpoints/an_default.pth --arch alexnet --topk 3

python predict.py flower/test/1/image_06743.jpg checkpoints/dn_default.pth --topk 7 --labels cat_to_name.json



python train.py flowers/ --lr 0.0005 --checkpoint checkpoints/dn_default.pth --epochs 2 --device cuda --newcheckpoint checkpoints/dn_update.pth

python train.py flowers/ --lr 0.0005 --arch alexnet --epochs 4 --device cuda --newcheckpoint checkpoints/an1.pth --hidden_units 1000 --output_units 102