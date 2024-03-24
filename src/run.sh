#!/bash/sh
python train.py --fold 10 --class_weight "{0: 1, 1: 1}" --model "LogisticRegression" --penalty "l2" 

