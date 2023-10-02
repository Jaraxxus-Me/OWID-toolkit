#!/bin/sh

while true
do
for ((scene_id=31560;scene_id<=50000;scene_id++))
do
blenderproc run train_set/render_p2_abo.py --scenes_id $scene_id
done
done
