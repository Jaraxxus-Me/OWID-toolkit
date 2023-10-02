import blenderproc as bproc
import debugpy
import os
import numpy as np
import argparse
from blenderproc.python.types.MeshObjectUtility import MeshObject
from typing import List
import pickle as pkl
import json
from blenderproc.python.loader.ObjectLoader import load_obj

parser = argparse.ArgumentParser()
parser.add_argument('--cc_textures_path', default="/home/airlab/storage/airlab/cc_textures", help="Path to downloaded cc textures")
parser.add_argument('--name_id_mapper_path', default="/home/airlab/storage/airlab/data_0108/data/InsDet/OWID/P1/ZID-10k_mapper_new.json", help="Path to name id mapper")
parser.add_argument('--shapenet_path', default="/home/airlab/storage/airlab/ShapeNetCore.v2", help="Path to the downloaded shape net core v2 dataset, get it from http://www.shapenet.org/")
parser.add_argument('--abo_path', default='/home/airlab/storage/airlab/abo-3dmodels/3dmodels/original', help="Path to the downloaded ABO, get it from https://amazon-berkeley-objects.s3.amazonaws.com/index.html")
parser.add_argument('--output_dir', default="/home/airlab/storage/airlab/OWID/P2_raw_2_abo", help="Path to where the final files, will be saved")
parser.add_argument('--num_objects', type=list, default=[8, 15], help="How many objects are in a scene")
parser.add_argument('--num_imgs', type=int, default=1, help="How many images of a scene to generate")
parser.add_argument('--scenes_id', type=int, default=0, help="How many scenes to generate")

bproc.init()
# debugpy.listen(5678)
# debugpy.wait_for_client()

def load_shapnet_objs(shapenet_dataset_path: str, abo_dataset_path: str, num_objs, name_id_mapper=None) -> List[MeshObject]:
    """ Loads all or a subset of 3D models of any BOP dataset
    :return: The list of loaded mesh objects.
    """
    mapper = name_id_mapper.copy()
    loaded_obj= []
    volume = []
    while len(loaded_obj) < num_objs:
        name = np.random.choice(list(mapper.keys()))
        rand_key, rand_v = name.split("_")[0], name.split("_")[1]
        if len(rand_key)>1: # shapenet
            random_obj = bproc.loader.load_shapenet(shapenet_dataset_path, used_synset_id=rand_key, used_source_id=rand_v)
            random_obj.set_cp("category_id", mapper[name])
            volume.append(random_obj.get_bound_box_volume())
            loaded_obj.append(random_obj)
        else:
            obj_path = os.path.join(abo_dataset_path, rand_key, '{}.glb'.format(rand_v))
            random_obj = load_obj(obj_path)[0]
            random_obj.set_cp("category_id", mapper[name])
            volume.append(random_obj.get_bound_box_volume())
            loaded_obj = [random_obj] + loaded_obj
        mapper.pop(name)
    volume_avg = np.mean(volume)
    volume_std = np.std(volume)
    rescale_obj = []
    for obj in loaded_obj:
        obj_volume = obj.get_bound_box_volume()
        if (obj_volume>volume_avg+volume_std) or (obj_volume>volume_avg-volume_std):
            scale_factor = (volume_avg/obj_volume)**(1/3)
            obj.set_scale([scale_factor, scale_factor, scale_factor])
        rescale_obj.append(obj)
    return rescale_obj

def main(s, args):
    orientation_info = {}
    # load the ShapeNet object into the scene
    if os.path.isfile(os.path.join(args.output_dir, "{:06d}".format(s), "coco_annotations.json")):
        print('skipping {}'.format(os.path.join(args.output_dir, "{:06d}".format(s))))
        return
    num_objects = np.random.randint(args.num_objects[0], args.num_objects[1])

    with open(args.name_id_mapper_path, 'r') as f:
        mapper = json.load(f)
    shapenet_objs = load_shapnet_objs(args.shapenet_path, args.abo_path, num_objects, mapper)


    # load BOP datset intrinsics
    bproc.loader.load_bop_intrinsics(bop_dataset_path = '/home/airlab/SONY2022/BOP_data/ycbv')

    # set shading and physics properties and randomize PBR materials
    for j, obj in enumerate(shapenet_objs):
        obj.set_shading_mode('auto')
            
        mat = obj.get_materials()[0]       
        mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
        mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))
            
    # create room
    room_planes = [bproc.object.create_primitive('PLANE', scale=[8, 8, 4]),
                bproc.object.create_primitive('PLANE', scale=[8, 8, 4], location=[0, -8, 8], rotation=[-1.570796, 0, 0]),
                bproc.object.create_primitive('PLANE', scale=[8, 8, 4], location=[0, 8, 8], rotation=[1.570796, 0, 0]),
                bproc.object.create_primitive('PLANE', scale=[8, 8, 4], location=[8, 0, 8], rotation=[0, -1.570796, 0]),
                bproc.object.create_primitive('PLANE', scale=[8, 8, 4], location=[-8, 0, 8], rotation=[0, 1.570796, 0])]

    # sample light color and strenght from ceiling
    light_plane = bproc.object.create_primitive('PLANE', scale=[6, 6, 2], location=[0, 0, 10])
    light_plane.set_name('light_plane')
    light_plane_material = bproc.material.create('light_material')
    light_plane_material.make_emissive(emission_strength=np.random.uniform(6,12), 
                                    emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))    
    light_plane.replace_materials(light_plane_material)

    # sample point light on shell
    light_point = bproc.types.Light()
    light_point.set_energy(150)
    light_point.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
    location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 4, radius_max = 6,
                            elevation_min = 5, elevation_max = 89, uniform_volume = False)
    light_point.set_location(location)

    # sample CC Texture and assign to room planes
    cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)
    random_cc_texture = np.random.choice(cc_textures)
    for plane in room_planes:
        plane.replace_materials(random_cc_texture)

    # Define a function that samples the initial pose of a given object above the ground
    def sample_initial_pose(obj: bproc.types.MeshObject):
        obj.set_location(bproc.sampler.upper_region(objects_to_sample_on=room_planes[0:1],
                                                    min_height=1, max_height=4, face_sample_range=[0.4, 0.6]))
        obj.set_rotation_euler(np.random.uniform([0, 0, 0], [np.pi/2, np.pi/2, np.pi * 2]))

    # Sample objects on the given surface
    placed_objects = bproc.object.sample_poses_on_surface(objects_to_sample=shapenet_objs,
                                            surface=room_planes[0],
                                            sample_pose_func=sample_initial_pose,
                                            min_distance=0.7,
                                            max_distance=0.9)

    # BVH tree used for camera obstacle checks
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(placed_objects)

    for poses in range(args.num_imgs):
        # Sample location
        location = bproc.sampler.shell(center = [0, 0, 0],
                                radius_min = 4.2,
                                radius_max = 4.8,
                                elevation_min = 5,
                                elevation_max = 70,
                                uniform_volume = False)
        # Determine point of interest in scene as the object closest to the mean of a subset of objects
        poi = bproc.object.compute_poi(np.random.choice(placed_objects, size=num_objects))
        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        
        # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
        if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
            # Persist camera pose
            # obj.get_local2world_mat()
            bproc.camera.add_camera_pose(cam2world_matrix)
            orientation_info['camera_w'] = cam2world_matrix
    for obj in placed_objects:
        orientation_info['{}_w'.format(obj.get_cp('category_id'))]=obj.get_local2world_mat()

    # render the whole pipeline
    bproc.renderer.set_max_amount_of_samples(50)
    bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"],default_values={'category_id': -1})
    data = bproc.renderer.render()
    bproc.writer.write_coco_annotations(os.path.join(args.output_dir, '{:06d}'.format(s)),
                                        instance_segmaps=data["instance_segmaps"],
                                        instance_attribute_maps=data["instance_attribute_maps"],
                                        colors=data["colors"],
                                        color_file_format="JPEG")
    with open(os.path.join(args.output_dir, '{:06d}'.format(s), 'orientation.pkl'), 'wb') as f:
        pkl.dump(orientation_info, f)

if __name__ == "__main__":
    args = parser.parse_args()
    print("Command Line Args:", args)
    main(
        args.scenes_id,
        args
    )

