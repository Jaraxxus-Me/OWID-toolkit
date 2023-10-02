#!/bin/sh

while true
do
for ((scene_id=56571;scene_id<=75000;scene_id++))
do
blenderproc run train_set/render_p2_abo.py --scenes_id $scene_id
done
done
