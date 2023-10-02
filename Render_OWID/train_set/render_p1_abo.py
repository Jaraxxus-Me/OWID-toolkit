import blenderproc as bproc
import argparse
import debugpy
import glob, os
import numpy as np
from blenderproc.python.loader.ObjectLoader import load_obj
from mathutils import Vector
# from tqdm import tqdm
# path config

bproc.init()
# debugpy.listen(5678)
# debugpy.wait_for_client()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--abo_path', default='/home/airlab/storage/airlab/abo-3dmodels/3dmodels/original', help="Path to the downloaded ABO, get it from https://amazon-berkeley-objects.s3.amazonaws.com/index.html")
    parser.add_argument('--output_dir', default="/home/airlab/storage/airlab/OWID/P1_raw_abo", help="Path to where the final files, will be saved")
    parser.add_argument('--synset_id', default="0", help="folder id to be rendered")
    parser.add_argument('--cc_textures_path', default="/home/airlab/storage/airlab/cc_textures", help="Path to downloaded cc textures")
    parser.add_argument('--num_imgs', type=int, default=40, help="How many images of a video to generate")
    parser.add_argument('--num_videos', type=int, default=1, help="How many videos of an object to generate")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    all_source_id = glob.glob(os.path.join(args.abo_path, args.synset_id, '*.glb'))
    skipped_id = ['B075HWDSDK', 'B07BG69MRW']

    # if args.synset_id in skipped_id:
    #     print('skipped {}'.format(args.synset_id))
    #     return

    bproc.loader.load_bop_intrinsics(bop_dataset_path = '/home/airlab/SONY2022/BOP_data/ycbv')

    def bmesh_from_object_final(ob):
        import bmesh
        matrix = ob.matrix_world
        me = ob.to_mesh()
        me.transform(matrix)
        bm = bmesh.new()
        bm.from_mesh(me)
        return (bm, matrix.is_negative)

    def volume_and_area_from_object(ob):
        bm, is_negative = bmesh_from_object_final(ob)
        volume = bm.calc_volume(signed=True)
        bm.free()
        if is_negative:
            volume = -volume
        return volume

    # create room
    room_planes = [bproc.object.create_primitive('PLANE', scale=[8, 8, 4]),
                bproc.object.create_primitive('PLANE', scale=[8, 8, 4], location=[0, -8, 8], rotation=[-1.570796, 0, 0]),
                bproc.object.create_primitive('PLANE', scale=[8, 8, 4], location=[0, 8, 8], rotation=[1.570796, 0, 0]),
                bproc.object.create_primitive('PLANE', scale=[8, 8, 4], location=[8, 0, 8], rotation=[0, -1.570796, 0]),
                bproc.object.create_primitive('PLANE', scale=[8, 8, 4], location=[-8, 0, 8], rotation=[0, 1.570796, 0])]

    # sample light color and strenght from ceiling
    light_plane = bproc.object.create_primitive('PLANE', scale=[4, 4, 1], location=[0, 0, 4])
    light_plane.set_name('light_plane')
    light_plane.set_cp("category_id", -1)
    light_plane_material = bproc.material.create('light_material')

    # sample point light on shell
    light_point = bproc.types.Light()
    light_point.set_energy(150)

    # load cc_textures
    cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)

    # Define a function that samples 6-DoF poses
    def sample_pose_func(obj: bproc.types.MeshObject):
        min = np.random.uniform([-0.01, -0.01, 0.2], [-0.02, -0.02, 0.2])
        max = np.random.uniform([0.02, 0.02, 0.5], [0.03, 0.03, 0.5])
        obj.set_location(np.random.uniform(min, max))

    # activate depth rendering without antialiasing and set amount of samples for color rendering
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_max_amount_of_samples(50)

    for source_id in all_source_id:
    # load the ABO object into the scene
        name = source_id.split('/')[-1][:-4]
        if os.path.isdir(os.path.join(args.output_dir, args.synset_id, name+'_00', 'test_video')):
            print('Finished {}'.format(name))
            continue
        if name in skipped_id:
            print('Skipped {}'.format(name))
            continue
        obj_path = source_id
        abo_obj = load_obj(obj_path)

        # Randomize materials and set physics
        for obj in (abo_obj):        
            # mat = obj.get_materials()[0]       
            # mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
            # mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))
            obj.hide(False)
            obj.set_cp("category_id", 0)

        # create videos:
        for s in range(args.num_videos):
            # Sample two light sources
            light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), 
                                            emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))  
            light_plane.replace_materials(light_plane_material)
            light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
            location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 2, radius_max = 2.5,
                                    elevation_min = 5, elevation_max = 89)
            light_point.set_location(location)

            # sample CC Texture and assign to room planes
            random_cc_texture = np.random.choice(cc_textures)
            for plane in room_planes:
                plane.replace_materials(random_cc_texture)
                plane.set_cp("category_id", -1)

            # Sample object poses and check collisions 
            bproc.object.sample_poses(objects_to_sample = abo_obj,
                                sample_pose_func = sample_pose_func, 
                                max_tries = 1000)

            # Define a function that samples the initial pose of a given object above the ground
            def sample_initial_pose(obj: bproc.types.MeshObject):
                min = np.random.uniform([-0.01, -0.01, 0.0], [-0.02, -0.02, 0.0])
                max = np.random.uniform([0.02, 0.02, 0.04], [0.03, 0.03, 0.06])
                obj.set_location(np.random.uniform(min, max))
                obj.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, np.pi * 2]))

            # Sample objects on the given surface
            placed_objects = bproc.object.sample_poses_on_surface(objects_to_sample=abo_obj,
                                                                surface=room_planes[0],
                                                                sample_pose_func=sample_initial_pose,
                                                                min_distance=0.01,
                                                                max_distance=0.2)

            # BVH tree used for camera obstacle checks
            bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(abo_obj)
            obj = abo_obj[0].blender_obj
            v = abs(volume_and_area_from_object(obj))
            obj_sz = pow(v, 1/3)
            camera_min = min(20 * obj_sz, 2.4)
            camera_max = min(32 * obj_sz, 2.8)
            # define a function to sample camera poses
            def location_circle(init_location, center_loc, N):
                r = np.linalg.norm(init_location[:-1] - center_loc[:-1])
                theta = np.arange(0, 2*np.pi, 2*np.pi/N)
                out=[]
                for the in theta:
                    dx = r*np.cos(the)
                    dy = r*np.sin(the)
                    out.append(np.array([center_loc[0]+dx*r, center_loc[1]+dy*r, init_location[2]]))
                return out
            # Sample initial camera location
            if obj.dimensions.z<0.5:
                camloc_init = bproc.sampler.shell(center = [0, 0, 0],
                                        radius_min = camera_min,
                                        radius_max = camera_max,
                                        elevation_min = 40,
                                        elevation_max = 50)
            else:
                camloc_init = bproc.sampler.shell(center = [0, 0, 0],
                                        radius_min = camera_min,
                                        radius_max = camera_max,
                                        elevation_min = 50,
                                        elevation_max = 60)

            # Determine point of interest in scene as the object closest to the mean of a subset of objects
            poi = bproc.object.compute_poi(np.random.choice(placed_objects, size=1, replace=False))
            # poi = np.array([0, 0, 0])
            cam_locs = location_circle(camloc_init, poi, args.num_imgs)
            cam_poses = 0
            for i in range(args.num_imgs):
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
            seg_data = bproc.renderer.render_segmap(map_by = ["class", "instance", "name"])
            origin_seg = seg_data['instance_segmaps']
            obj_name = abo_obj[0].blender_obj.name
            for dic in seg_data['instance_attribute_maps'][0]:
                if dic['name'] == abo_obj[0].blender_obj.name:
                    obj_id = dic['idx']
            seg_map = []
            for n in range(len(origin_seg)):
                one_map = np.zeros_like(origin_seg[n])
                one_map[origin_seg[n]==obj_id] = 1
                seg_map.append(one_map)
            # Write data in bop format
            bproc.writer.write_air(os.path.join(args.output_dir, args.synset_id),
                                target_objects = abo_obj,
                                dataset = '{}_{:02d}'.format(name, s),
                                depth_scale = 0.1,
                                depths = data["depth"],
                                colors = data["colors"],
                                semantic = seg_map,
                                color_file_format = "JPEG",
                                ignore_dist_thres = 10)

        for obj in (abo_obj):      
            obj.hide(True)

if __name__ == '__main__':
    main()
