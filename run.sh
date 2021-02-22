identify style/vassily.jpg |grep -o "[[:digit:]]*x[[:digit:]]*" |head -1
python train.py --dataset VOCdevkit/VOC2007/JPEGImages/ --style_image van.jpg --res 400 --train_epoch 100  --save_root van_style  
python train.py --dataset VOCdevkit/VOC2007/JPEGImages/ --style_image van.jpg --res 128 --train_epoch 61  --save_root van_small  