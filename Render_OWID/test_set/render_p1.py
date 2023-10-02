import blenderproc as bproc
import argparse
import os
import numpy as np
import debugpy

parser = argparse.ArgumentParser()
parser.add_argument('--bop_parent_path', default = '/home/airlab/SONY2022/BOP_data', help="Path to the bop datasets parent directory")
parser.add_argument('--cc_textures_path', default="/home/airlab/SONY2022/cc_textures_selected_test", help="Path to downloaded cc textures")
parser.add_argument('--output_dir', default = './output', help="Path to where the final files will be saved ")
parser.add_argument('--num_scenes', type=int, default=160, help="How many scenes of an object to generate")
parser.add_argument('--dataset', type=str, default='ycbv', help="Which bop dataset to render?")
parser.add_argument('--seed', type=int, default=17, help="random seed")

args = parser.parse_args()

# debugpy.listen(5678)
# debugpy.wait_for_client()

bproc.init()
np.random.seed(args.seed)
# load bop objects into the scene
target_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, args.dataset), mm2m = True)


# load BOP datset intrinsics
bproc.loader.load_bop_intrinsics(bop_dataset_path = '/home/airlab/SONY2022/BOP_data/ycbv')

# set shading and hide objects + tless_dist_bop_objs + ycbv_dist_bop_objs + tyol_dist_bop_objs
for obj in (target_bop_objs):
    obj.set_shading_mode('auto')
    obj.hide(True)
    
# create room
room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])]

# sample light color and strenght from ceiling
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 1])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')

# sample point light on shell
light_point = bproc.types.Light()
light_point.set_energy(150)

# load cc_textures
cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)

# Define a function that samples 6-DoF poses
def sample_pose_func(obj: bproc.types.MeshObject):
    min = np.random.uniform([-0.01, -0.01, 0.0], [-0.02, -0.02, 0.0])
    max = np.random.uniform([0.02, 0.02, 0.04], [0.03, 0.03, 0.06])
    obj.set_location(np.random.uniform(min, max))
    # obj.set_rotation_euler(bproc.sampler.uniformSO3())
    
# activate depth rendering without antialiasing and set amount of samples for color rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)
target_bop_objs = target_bop_objs[17:18]

for o in range(len(target_bop_objs)):
    # Sample bop objects for a scene
    sampled_target_bop_objs = [target_bop_objs[o]]

    # Randomize materials and set physics
    for obj in (sampled_target_bop_objs):        
        mat = obj.get_materials()[0]       
        mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
        mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))
        obj.hide(False)
    
    # Sample two light sources
    light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), 
                                    emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))  
    light_plane.replace_materials(light_plane_material)
    light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
    location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
                            elevation_min = 5, elevation_max = 89)
    light_point.set_location(location)

    # sample CC Texture and assign to room planes 0 2 3 4 5 10 16 59 78 88 100 126 150 250 293 493 450 364 523 546 593 785 786 923 980 620
    random_cc_texture = np.random.choice(cc_textures)
    for plane in room_planes:
        plane.replace_materials(random_cc_texture)

    # Sample object poses and check collisions 
    bproc.object.sample_poses(objects_to_sample = sampled_target_bop_objs,
                            sample_pose_func = sample_pose_func, 
                            max_tries = 1000)
            
    # Define a function that samples the initial pose of a given object above the ground
    def sample_initial_pose(obj: bproc.types.MeshObject):
        min = np.random.uniform([-0.01, -0.01, 0.0], [-0.02, -0.02, 0.0])
        max = np.random.uniform([0.02, 0.02, 0.04], [0.03, 0.03, 0.06])
        obj.set_location(np.random.uniform(min, max))
        obj.set_rotation_euler([0, 0, 0])

    # Sample objects on the given surface
    placed_objects = bproc.object.sample_poses_on_surface(objects_to_sample=sampled_target_bop_objs,
                                                        surface=room_planes[0],
                                                        sample_pose_func=sample_initial_pose,
                                                        min_distance=0.01,
                                                        max_distance=0.2)

    # BVH tree used for camera obstacle checks
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_target_bop_objs)

    cam_poses = 0
    def location_circle(init_location, center_loc, N):
        r = np.linalg.norm(init_location[:-1] - center_loc[:-1])
        theta = np.arange(0, 2*np.pi, 2*np.pi/N)
        out=[]
        for the in theta:
            dx = r*np.cos(the)
            dy = r*np.sin(the)
            out.append(np.array([center_loc[0]+dx*r, center_loc[1]+dy*r, init_location[2]]))
        return out
            # Sample location
    camloc_init = bproc.sampler.shell(center = [0, 0, 0],
                            radius_min = 0.6,
                            radius_max = 0.8,
                            elevation_min = 15,
                            elevation_max = 25)
    
    # Determine point of interest in scene as the object closest to the mean of a subset of objects
    poi = bproc.object.compute_poi(np.random.choice(sampled_target_bop_objs, size=1, replace=False))
    cam_locs = location_circle(camloc_init, [0, 0, 0], args.num_scenes)
    for i in range(args.num_scenes):
        # Sample location
        location = cam_locs[i]
        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location)
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        
        # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
        if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.02}, bop_bvh_tree):
            # Persist camera pose
            bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
            cam_poses += 1

    # render the whole pipeline
    data = bproc.renderer.render()
    seg_data = bproc.renderer.render_segmap(map_by = ["instance", "name"])
    origin_seg = seg_data['instance_segmaps']
    obj_name = sampled_target_bop_objs[0].blender_obj.name
    for dic in seg_data['instance_attribute_maps'][0]:
        if dic['name'] == sampled_target_bop_objs[0].blender_obj.name:
            obj_id = dic['idx']
    seg_map = []
    for n in range(len(origin_seg)):
        one_map = np.zeros_like(origin_seg[n])
        one_map[origin_seg[n]==obj_id] = 1
        seg_map.append(one_map)
    # Write data in bop format
    bproc.writer.write_air(os.path.join(args.output_dir, 'bop_data'),
                        target_objects = sampled_target_bop_objs,
                        dataset = args.dataset,
                        depth_scale = 0.1,
                        depths = data["depth"],
                        colors = data["colors"],
                        semantic = seg_map,
                        color_file_format = "JPEG",
                        ignore_dist_thres = 10)
    
    for obj in (sampled_target_bop_objs):      
        obj.hide(True)