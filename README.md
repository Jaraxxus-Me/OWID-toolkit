# OWID-toolkit

This library includes code to generate the synthetic training set, [OWID](https://drive.google.com/file/d/14eXttq02c8fGHt5nLlFhnjVrbSk3kQQn/view?usp=sharing), which is for [VoxDet](https://github.com/Jaraxxus-Me/VoxDet) training.

You can easily customize your own instance dataset using this toolkit for higher-quality 3D CAD models!

The code consists of two parts `Render_OWID` is a set of tool scripts that use the functions in BlenderProc to load 3D models and render images. `BlenderProc` is a modified version of the public BlenderProc2 library to better fit our needs

Please follow the steps to render the dataset.

For reference, this code is built upon the tutorial [here](https://dlr-rm.github.io/BlenderProc/examples/datasets/scenenet_with_cctextures/README.html), you can also have a look for deeper understanding.



## Step1: Download the following datasets
- [ShapeNet](https://shapenet.org/account/), we used the ShapeNetCore_v1 dataset.
- [ABO](https://amazon-berkeley-objects.s3.amazonaws.com/index.html#download), we used the above-3dmodels.tar
- cc_texture, please follow the description in the [tutorial](https://dlr-rm.github.io/BlenderProc/examples/datasets/scenenet_with_cctextures/README.html)





## Step2: Install Blenderproc2
```shell
cd BlenderProc
pip install -e .
```



## Step3: Render p1 data

```shell
cd Render_OWID
bash render_p1_0.sh # please modify the directions according to your ABO and ShapeNet path
```



## Step4: Format p1 data

```shell
cd train_set
python3 format_p1.py # please modify the directions accordingly
```

This formats p1 data into a structured folder and also generates the mapping dictionary for p2 rendering.



## Step5: Render p2 data

```shell
cd ..
bash render_p2_0.sh # you can open 4 tmux windows and run the render_p2_0,1,2,3.sh in parallel, please modify the directions accordingly
```



## Step6: Format and split p2 data

```shell
cd train_set
python3 format_p2.py # please modify the directions accordingly
python3 split_p2.py # please modify the directions accordingly, you can also set a custom train/val ratio
```



**Note**: In `Render_OWID/test_set`, we also provide the script to render a 360 video of the test instances, it follows a similar procedure for the training set. 



## Acknowledgment

The authors sincerely thank the developers of [Blenderproc2](https://github.com/DLR-RM/BlenderProc), we build our toolkit upon their code library and tutorial. The authors would also like to thank the authors of [ShapeNet](https://shapenet.org/account/) and [ABO](https://amazon-berkeley-objects.s3.amazonaws.com/index.html#download) for providing high-quality 3D models.

If our toolkit helps your research, please cite us as:

```
@INPROCEEDINGS{Li2023Vox,       
	author={Li, Bowen and Wang, Jiashun and Hu, Yaoyu and Wang, Chen and Scherer, Sebastian},   
	booktitle={Proceedings of the Advances in Neural Information Processing Systems (NeurIPS)}, 
	title={{VoxDet: Voxel Learning for Novel Instance Detection}},
	year={2023},
	volume={},
	number={}
}
```



