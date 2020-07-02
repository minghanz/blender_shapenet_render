""" render_rgb.py renders obj file to rgb image

Aviable function:
- clear_mash: delete all the mesh in the secene
- scene_setting_init: set scene configurations
- node_setting_init: set node configurations
- render: render rgb image for one obj file and one viewpoint
- render_obj_by_vp_lists: wrapper function for render() render 
                          one obj file by multiple viewpoints
- render_objs_by_one_vp: wrapper function for render() render
                         multiple obj file by one viewpoint
- init_all: a wrapper function, initialize all configurations                          
= set_image_path: reset defualt image output folder

author baiyu
"""
import sys
import os
import random
import pickle
import bpy
import bmesh

from mathutils import Matrix
from mathutils import Vector

abs_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(abs_path))

from render_helper import *
from settings import *
import settings

from vp_generator import get_focal_len_from_obj_and_pose, get_bot_pts, project_pts_to_cam, get_4x4_RT_matrix_from_blender, get_K_from_blender, get_2d_bbox, write_cam_pose

import time

### preload all background image paths
background_img_list = gen_list_of_valid_background_img()

def clear_mesh():
    """ clear all meshes in the secene

    """
    ### Added to handle GPU memory leak
    ### https://blender.stackexchange.com/a/102046
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)

    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)

    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)

    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)
    ##################################
    
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
            bpy.ops.object.delete()


def scene_setting_init(use_gpu):
    """initialize blender setting configurations

    """
    sce = bpy.context.scene.name
    bpy.data.scenes[sce].render.engine = g_engine_type
    bpy.data.scenes[sce].render.film_transparent = g_use_film_transparent
    #output
    bpy.data.scenes[sce].render.image_settings.color_mode = g_rgb_color_mode
    bpy.data.scenes[sce].render.image_settings.color_depth = g_rgb_color_depth
    bpy.data.scenes[sce].render.image_settings.file_format = g_rgb_file_format

    #dimensions
    bpy.data.scenes[sce].render.resolution_x = g_resolution_x
    bpy.data.scenes[sce].render.resolution_y = g_resolution_y
    bpy.data.scenes[sce].render.resolution_percentage = g_resolution_percentage

    if use_gpu:
        bpy.data.scenes[sce].render.engine = 'CYCLES' #only cycles engine can use gpu
        bpy.data.scenes[sce].render.tile_x = g_hilbert_spiral
        bpy.data.scenes[sce].render.tile_x = g_hilbert_spiral
        bpy.context.scene.cycles.device = 'GPU'
        bpy.data.scenes[sce].cycles.device = 'GPU'

def node_setting_init():
    """node settings for render rgb images

    mainly for compositing the background images
    """

    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    for node in tree.nodes:
        tree.nodes.remove(node)
    
    render_layer_node = tree.nodes.new('CompositorNodeRLayers')
    file_output_node = tree.nodes.new('CompositorNodeOutputFile')

    # ### if we want to render image with background
    # image_node = tree.nodes.new('CompositorNodeImage')
    # scale_node = tree.nodes.new('CompositorNodeScale')
    # alpha_over_node = tree.nodes.new('CompositorNodeAlphaOver')

    # scale_node.space = g_scale_space
    # file_output_node.base_path = g_syn_rgb_folder
    # if not os.path.exists(file_output_node.base_path):
    #     os.makedirs(file_output_node.base_path)

    # links.new(image_node.outputs[0], scale_node.inputs[0])
    # links.new(scale_node.outputs[0], alpha_over_node.inputs[1])
    # links.new(render_layer_node.outputs[0], alpha_over_node.inputs[2])
    # links.new(alpha_over_node.outputs[0], file_output_node.inputs[0])

    ### render black background image
    file_output_node.base_path = g_syn_rgb_folder
    if not os.path.exists(file_output_node.base_path):
        os.makedirs(file_output_node.base_path)

    links.new(render_layer_node.outputs[0], file_output_node.inputs[0]) # render_layer_node [0] is image, [1] is alpha (binary mask), [2] is depth

    #### render silhouette output image here
    #### https://blender.stackexchange.com/questions/42579/render-depth-map-to-image-with-python-script
    # map_range_node_2 = tree.nodes.new('CompositorNodeMapRange')
    # map_range_node_2.inputs[1].default_value = 1
    # map_range_node_2.inputs[2].default_value = 100
    # map_range_node_2.inputs[3].default_value = 1
    # map_range_node_2.inputs[4].default_value = 10
    
    file_output_node_2 = tree.nodes.new('CompositorNodeOutputFile')
    file_output_node_2.base_path = g_syn_depth_folder
    if not os.path.exists(file_output_node_2.base_path):
        os.makedirs(file_output_node_2.base_path)

    # links.new(render_layer_node.outputs[2], map_range_node_2.inputs[0])
    # links.new(map_range_node_2.outputs[0], file_output_node_2.inputs[0])
    links.new(render_layer_node.outputs[1], file_output_node_2.inputs[0]) # render_layer_node [0] is image, [1] is alpha (binary mask), [2] is depth


def render(obj_path, viewpoint):
    """render rbg image 

    render a object rgb image by a given camera viewpoint and
    choose random image as background, only render one image
    at a time.

    Args:
        obj_path: a string variable indicate the obj file path
        viewpoint: a vp parameter(contains azimuth,elevation,tilt angles and distance)
    """

    current_frame = bpy.context.scene.frame_current
    file_output_node = bpy.context.scene.node_tree.nodes[4]
    
    ### gen txt file with the same name as the output image
    fpath = os.path.join(file_output_node.base_path, 'blender-{:06d}.color.txt'.format(current_frame))
    open(fpath, 'w').close()

    ### random sample a background image
    image_path = random.choice(background_img_list)
    vp = viewpoint

    with open(fpath, 'a') as f:
        f.write("model_path: {}\n".format(obj_path) )
        f.write("bg_image_path: {}\n".format(image_path))
        f.write("vp: {} {} {} {}\n".format(vp.azimuth, vp.elevation, vp.tilt, vp.distance))

    ### calculate camera pose from viewpoint
    cam_location = camera_location(vp.azimuth, vp.elevation, vp.distance)
    cam_rot = camera_rot_XYZEuler(vp.azimuth, vp.elevation, vp.tilt)

    cam_obj = bpy.data.objects['Camera']
    cam_obj.location[0] = cam_location[0]
    cam_obj.location[1] = cam_location[1]
    cam_obj.location[2] = cam_location[2]

    cam_obj.rotation_euler[0] = cam_rot[0]
    cam_obj.rotation_euler[1] = cam_rot[1]
    cam_obj.rotation_euler[2] = cam_rot[2]

    bpy.context.view_layer.update() # this is necessary for the change to be applied on cam_obj.matrix_world

    ### find the proper focal length to preserve the size of the object in image roughly
    for ob in bpy.context.scene.objects:
        if ob.type == 'MESH' and 'Basic_Sphere' not in ob.name:
            obj_target = ob
    new_len = get_focal_len_from_obj_and_pose(obj_target, cam_obj, view_dist=vp.distance, fpath=fpath)

    bpy.context.view_layer.update()

    ### get the feature points (contact point of tires with the ground)
    p_output = get_bot_pts(obj_target, fpath=fpath)
    # ### add a small ball at the position of feature points
    # for i in range(p_output.shape[1]):
    #     gen_sphere_from_pts( p_output[:,i] )

    ### project feature points to the image coordinate
    cam_pos = get_4x4_RT_matrix_from_blender(cam_obj)
    K_homo = get_K_from_blender(cam_obj, bpy.context.scene, new_len)
    write_cam_pose(cam_pos, K_homo, fpath)
    project_pts_to_cam(p_output, cam_pos, K_homo, fpath=fpath)

    ### 2D bbox 
    get_2d_bbox(obj_target, cam_pos, K_homo, fpath=fpath)

    if not os.path.exists(g_syn_rgb_folder):
        os.mkdir(g_syn_rgb_folder)

    ### load the background image
    image_node = bpy.context.scene.node_tree.nodes[0]
    image_node.image = bpy.data.images.load(image_path)

    ### set the output file name
    file_output_node.file_slots[0].path = 'blender-######.color.png' # blender placeholder #

    ### start rendering
    bpy.ops.render.render(write_still=True)
    bpy.context.scene.frame_set(current_frame + 1)


def render_obj_by_vp_lists(obj_path, viewpoints):
    """ render one obj file by a given viewpoint list
    a wrapper function for render()

    Args:
        obj_path: a string variable indicate the obj file path
        viewpoints: an iterable object of vp parameter(contains azimuth,elevation,tilt angles and distance)
    """

    if isinstance(viewpoints, tuple):
        vp_lists = [viewpoints]

    try:
        vp_lists = iter(viewpoints)
    except TypeError:
        print("viewpoints is not an iterable object")
    
    for vp in vp_lists:
        render(obj_path, vp)

def render_objs_by_one_vp(obj_pathes, viewpoint):
    """ render multiple obj files by a given viewpoint

    Args:
        obj_paths: an iterable object contains multiple
                   obj file pathes
        viewpoint: a namedtuple object contains azimuth,
                   elevation,tilt angles and distance
    """ 

    if isinstance(obj_pathes, str):
        obj_lists = [obj_pathes]
    
    try:
        obj_lists = iter(obj_lists)
    except TypeError:
        print("obj_pathes is not an iterable object")
    
    for obj_path in obj_lists:
        render(obj_path, viewpoint)

def init_all():
    """init everything we need for rendering
    an image
    """
    scene_setting_init(g_gpu_render_enable)
    node_setting_init()
    cam_obj = bpy.data.objects['Camera']
    cam_obj.rotation_mode = g_rotation_mode

    bpy.data.objects['Light'].data.energy = 100
    bpy.ops.object.light_add(type='SUN')

def set_image_path(new_path):
    """ set image output path to new_path

    Args:
        new rendered image output path
    """
    file_output_node = bpy.context.scene.node_tree.nodes[4]
    file_output_node.base_path = new_path

def combine_objects():
    """combine all objects in the scene
    """
    scene = bpy.context.scene
    obs = []

    for ob in scene.objects:
    # whatever objects you want to join...
        if ob.type == 'MESH':
            obs.append(ob)

    ctx = bpy.context.copy()
    # one of the objects to join
    ctx['active_object'] = obs[0]
    ctx['selected_objects'] = obs
    # we need the scene bases as well for joining
    ctx['selected_editable_bases'] = [scene.object_bases[ob.name] for ob in obs]
    bpy.ops.object.join(ctx)

def scale_objects(scale_factor):
    """Scale all mesh objects in the scene, use combine_objects before this
    function
    Args:
        scale_factor: scale percentage
    """
    scene = bpy.context.scene

    for ob in scene.objects:
        ob.select = False
        if ob.type == 'MESH':
            bpy.context.scene.objects.active = ob

    obj = bpy.context.scene.objects.active
    obj.scale = (scale_factor, scale_factor, scale_factor)


def get_3x4_RT_matrix_from_blender_nnp(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    # NOTE: Use * instead of @ here for older versions of Blender
    # TODO: detect Blender version
    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
         ))

    return RT


def gen_sphere_from_pts(position):
    """Create a sphere object at the given position
    https://blender.stackexchange.com/questions/93298/create-a-uv-sphere-object-in-blender-from-python"""
    # Create an empty mesh and the object.
    mesh = bpy.data.meshes.new('Basic_Sphere')
    basic_sphere = bpy.data.objects.new("Basic_Sphere", mesh)

    # Add the object into the scene.
    bpy.context.collection.objects.link(basic_sphere)

    # Select the newly created object
    bpy.context.view_layer.objects.active = basic_sphere
    basic_sphere.select_set(True)

    # Construct the bmesh sphere and assign it to the blender mesh.
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=4, diameter=0.01)
    bm.to_mesh(mesh)
    bm.free()

    bpy.ops.object.modifier_add(type='SUBSURF')
    bpy.ops.object.shade_smooth()

    basic_sphere.location[0] = position[0]
    basic_sphere.location[1] = position[1]
    basic_sphere.location[2] = position[2]

    return

### YOU CAN WRITE YOUR OWN IMPLEMENTATION TO GENERATE DATA

if __name__ == "__main__":
    start_time = time.time()
    init_all()

    result_dict = pickle.load(open(os.path.join(g_temp, g_result_dict), 'rb'))
    result_list = [result_dict[name] for name in g_render_objs]

    for obj_name, models in zip(g_render_objs, result_list):
        obj_folder = os.path.join(g_syn_rgb_folder, obj_name)
        if not os.path.exists(obj_folder):
            os.makedirs(obj_folder)
        
        print("number of models: ", len(models))
        for model in models:
            clear_mesh()
            bpy.ops.import_scene.obj(filepath=model.path)
            #combine_objects()
            #scale_objects(0.5)
            set_image_path(obj_folder)
            render_obj_by_vp_lists(model.path, model.vps)

    end_time = time.time()
    print("Time:", end_time - start_time)