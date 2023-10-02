#!/bin/sh
while true
    do
    for dir in `ls /home/airlab/storage/airlab/abo-3dmodels/3dmodels/original`; do
        echo "$dir"
        blenderproc run train_set/render_p1_abo.py --synset_id $dir
    done
done



while true
    do
    for dir in `ls /home/airlab/storage/airlab/ShapeNetCore.v2`; do
        echo "$dir"
        blenderproc run train_set/render_p1.py --synset_id $dir
    done
done