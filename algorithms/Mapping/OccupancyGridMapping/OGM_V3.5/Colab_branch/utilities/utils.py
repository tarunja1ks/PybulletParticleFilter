###############################################################################
# A very HUGE chunk of this code is inspired, modified, or copy pasted from the
# following repo(s):
#         @ GitHub: caelan/pybullet-planning/pybullet_tools/utils.py
###############################################################################
import os
import sys
import math
import time
import platform
import numpy as np
import pybullet as p
from collections import namedtuple
import repository.utilities.geometry_utils as gu

# For reading user configs (.yaml files)
import yaml
from yaml import Loader

CLIENTS = {}
CLIENT = 0
BASE_LINK = -1
STATIC_MASS = 0
NULL_ID = -1

CollisionShapeData = namedtuple(
        'CollisionShapeData',
        [
            'object_unique_id',
            'linkIndex',
            'geometry_type',
            'dimensions',
            'filename',
            'local_frame_pos',
            'local_frame_orn'
        ]
    )

AABB = namedtuple('AABB', ['lower', 'upper'])
LinkState = namedtuple('LinkState', ['linkWorldPosition', 'linkWorldOrientation',
                                     'localInertialFramePosition', 'localInertialFrameOrientation',
                                     'worldLinkFramePosition', 'worldLinkFrameOrientation'])
RGB = namedtuple('RGB', ['red', 'green', 'blue'])
RGBA = namedtuple('RGBA', ['red', 'green', 'blue', 'alpha'])
MAX_RGB = 2**8 - 1

RED = RGBA(1, 0, 0, 1)
GREEN = RGBA(0, 1, 0, 1)
BLUE = RGBA(0, 0, 1, 1)
BLACK = RGBA(0, 0, 0, 1)
WHITE = RGBA(1, 1, 1, 1)
BROWN = RGBA(0.396, 0.263, 0.129, 1)
TAN = RGBA(0.824, 0.706, 0.549, 1)
GREY = RGBA(0.5, 0.5, 0.5, 1)
YELLOW = RGBA(1, 1, 0, 1)
TRANSPARENT = RGBA(0, 0, 0, 0)

def load_configs(fname: str) -> dict:
    """
    Takes a .yaml file (str), reads
    and return the data contained as 
    a Python dictionary.

    Args:
        fname: Name of the .yaml file.
    Returns:
        A dictionary containing the data in the 
        file.
    Raises:
        AssertionError: if fname is not .yaml file.
    """
    assert fname.split('.')[-1] == "yaml"

    file = open(fname, 'r')
    return yaml.load(file, Loader=Loader)

def hex_to_rgba(hex_code):
    # Remove '#' symbol if present
    hex_code = hex_code.lstrip('#').lower()
    
    # Extract R, G, B values from hex code
    r, g, b = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
    # Set alpha value to 1
    a = 1.0
    
    # Normalize R, G, B values to range of 0 to 1
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0
    return RGBA(r_norm, g_norm, b_norm, a)

def add_line(start, end, color=BLACK, width=5, lifetime=None, parent=NULL_ID, parent_link=BASE_LINK):
    assert (len(start) == 3) and (len(end) == 3)
    return p.addUserDebugLine(start, end, lineColorRGB=color[:3], lineWidth=width,
                              lifeTime=get_lifetime(lifetime), parentObjectUniqueId=parent, parentLinkIndex=parent_link,
                              physicsClientId=CLIENT)

def get_lifetime(lifetime):
    if lifetime is None:
        return 0
    return lifetime

def point_from_pose(pose):
    return pose[0]

def multiply(*poses):
    pose = poses[0]
    for next_pose in poses[1:]:
        pose = p.multiplyTransforms(pose[0], pose[1], *next_pose)
    return pose

def tform_point(affine, point):
    return point_from_pose(multiply(affine, gu.Pose(point=point)))

def add_line(start, end, color=BLACK, width=1, lifetime=None, parent=NULL_ID, parent_link=BASE_LINK):
    assert (len(start) == 3) and (len(end) == 3)
    #time.sleep(1e-3) # When too many lines are added within a short period of time, the following error can occur
    return p.addUserDebugLine(start, end, lineColorRGB=color[:3], lineWidth=width,
                              lifeTime=get_lifetime(lifetime), parentObjectUniqueId=parent, parentLinkIndex=parent_link,
                              physicsClientId=CLIENT)

def draw_pose(pose, length=0.1, d=3, **kwargs):
    origin_world = tform_point(pose, np.zeros(3))
    handles = []
    for k in range(d):
        axis = np.zeros(3)
        axis[k] = 1
        axis_world = tform_point(pose, length*axis)
        handles.append(add_line(origin_world, axis_world, color=axis, **kwargs))
    return handles

def draw_robot_frame(car_id, baselink=BASE_LINK):
    """
    Draw the local coordinate frame (x, y, z axis)
    of the robot for debugging purposes.
    """
    handles = []
    for i in range(10):
        world_from_robot = get_link_pose(car_id, baselink)
        handles.extend(draw_pose(world_from_robot, length=0.5))

def remove_debug(debug):
    p.removeUserDebugItem(debug, physicsClientId=CLIENT)

def get_rotation_matrix(body):
    """
    Get the rotation matrix between the world frame and
    the `body` frame.
    """
    pos, quat = p.getBasePositionAndOrientation(body, physicsClientId=CLIENT)
    R = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    return R

def get_link_pose(body, link):
    if link == BASE_LINK:
        return get_pose(body)
    # if set to 1 (or True), the Cartesian world position/orientation will be recomputed using forward kinematics.
    link_state = get_link_state(body, link) #, kinematics=True, velocity=False)
    return link_state.worldLinkFramePosition, link_state.worldLinkFrameOrientation

def get_link_state(body, link, kinematics=True, velocity=True):
    # TODO: the defaults are set to False?
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/pybullet.c
    return LinkState(*p.getLinkState(body, link,
                                     #computeForwardKinematics=kinematics,
                                     #computeLinkVelocity=velocity,
                                     physicsClientId=CLIENT))

def get_xyzw_ori(body):
    '''
    Return orientation list (point) of 4 floats [x, y, z, w] of
    the `body`.
    '''
    return p.getBasePositionAndOrientation(body, physicsClientId=CLIENT)[1]

def get_xyz_point(body):
    '''
    Return position list (point) of 3 floats [x, y, z] of
    the `body`.
    '''
    return p.getBasePositionAndOrientation(body, physicsClientId=CLIENT)[0]

def get_pose(body):
    return p.getBasePositionAndOrientation(body, physicsClientId=CLIENT)

def euler_from_quat(quat):
    return p.getEulerFromQuaternion(quat) # rotation around fixed axis

def get_euler(body):
    ori = get_xyzw_ori(body) # Orientation in quaternion, [x, y, z, w]
    return p.getEulerFromQuaternion(ori)

def get_pitch(point):
    dx, dy, dz = point
    return np.math.atan2(dz, np.sqrt(dx ** 2 + dy ** 2))

def get_yaw(point):
    dx, dy = point[:2]
    return np.math.atan2(dy, dx)

def unit_pose():
    unit_point = (0., 0., 0.)
    unit_quat = p.getQuaternionFromEuler([0, 0, 0])
    return (unit_point, unit_quat)

def get_box_geometry(w, l, h):
    return {
        'shapeType': p.GEOM_BOX,
        'halfExtents': [w/2., l/2., h/2.]
    }

def get_floor_geometry(w, l, h, fileName, meshScale):
    return {
        'shapeType': p.GEOM_MESH,
        'halfExtents': [w/2., l/2., h/2.],
        'fileName': fileName,
        'meshScale': meshScale
    }

def get_aabb(body, link=None):
    '''
    TODO
    '''
    if link is None:
        return aabb_union(get_aabbs(body))

    return AABB(*p.getAABB(body, linkIndex=link, physicsClientId=CLIENT))

def get_aabbs(body, links=None, only_collision=True):
    '''
    TODO
    '''
    if links is None:
        links = get_all_links(body)
    if only_collision:
        links = [link for link in links if can_collide(body, link)]

    return [get_aabb(body, link=link) for link in links]

def get_aabb_center(aabb):
    lower, upper = aabb
    return (np.array(lower) + np.array(upper)) / 2.

def get_aabb_extent(aabb):
    lower, upper = aabb
    return np.array(upper) - np.array(lower)

def get_center_extent(body, **kwargs):
    aabb = get_aabb(body, **kwargs)
    return get_aabb_center(aabb), get_aabb_extent(aabb)

def get_all_links(body):
    '''
    Gets all links of a body (hence ).
    Args:
    Returns:
    Raises:
    '''
    links = list(range(p.getNumJoints(body, \
                                      physicsClientId=CLIENT)))
    return [BASE_LINK] + list(links)

def create_collision_shape(geometry, pose=unit_pose()):
    # TODO: removeCollisionShape
    # https://github.com/bulletphysics/bullet3/blob/5ae9a15ecac7bc7e71f1ec1b544a55135d7d7e32/examples/pybullet/examples/getClosestPoints.py
    point, quat = pose
    collision_args = {
        'collisionFramePosition': point,
        'collisionFrameOrientation': quat,
        'physicsClientId': CLIENT,
        #'flags': p.GEOM_FORCE_CONCAVE_TRIMESH,
    }
    collision_args.update(geometry)
    if 'length' in collision_args:
        # TODO: pybullet bug visual => length, collision => height
        collision_args['height'] = collision_args['length']
        del collision_args['length']
    return p.createCollisionShape(**collision_args)

def create_visual_shape(geometry, pose=unit_pose(), color=RED, specular=None):
    if (color is None): # or not has_gui():
        return NULL_ID
    point, quat = pose
    visual_args = {
        'rgbaColor': color,
        'visualFramePosition': point,
        'visualFrameOrientation': quat,
        'physicsClientId': CLIENT,
    }
    visual_args.update(geometry)
    if specular is not None:
        visual_args['specularColor'] = specular
    return p.createVisualShape(**visual_args)

def create_shape(geometry, pose=unit_pose(), collision=True, **kwargs):
    collision_id = create_collision_shape(geometry, pose=pose) if collision else NULL_ID
    visual_id = create_visual_shape(geometry, pose=pose, **kwargs) # if collision else NULL_ID
    return collision_id, visual_id

def create_body(collision_id=NULL_ID, visual_id=NULL_ID, mass=STATIC_MASS):
    return p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_id,
                             baseVisualShapeIndex=visual_id, physicsClientId=CLIENT)

def create_box(w, l, h, mass=STATIC_MASS, color=RED, **kwargs):
    '''
    Create an obstacle (box).
    '''
    collision_id, visual_id = create_shape(get_box_geometry(w, l, h), color=color, **kwargs)
    return create_body(collision_id, visual_id, mass=mass)

def create_floor(w, l, h, fileName, meshScale=[1., 1., 1.], mass=STATIC_MASS, color=RED, **kwargs):
    '''
    Create an obstacle (box).
    '''
    collision_id, visual_id = create_shape(get_floor_geometry(w, l, h, fileName, meshScale), color=color, **kwargs)
    return create_body(collision_id, visual_id, mass=mass)

def set_pose(body, pose):
    (point, quat) = pose
    p.resetBasePositionAndOrientation(body, point, quat, physicsClientId=CLIENT)

def set_point(body, point):
    '''
    Set the position of the obstacle.
    '''
    set_pose(body, (point, get_xyzw_ori(body)))

def set_quat(body, quat):
    set_pose(body, (get_xyz_point(body), quat))

def set_euler(body, euler):
    set_quat(body, p.getQuaternionFromEuler(euler))

def stable_z(body, surface, surface_link=None):
    '''
    Finds the appropiate z-offset so we 
    can place the robot on the 'floor'.
    Inspired by:
        credits:
        - caelan/pybullet-planning/pybullet_tools/utils.py
    My thorough explanation of how/why this function works:
        @[put link to google doc containing explanation here]
        (Draft)
            Basically it creates a box using pybullet function
            p.getAABB(). Then we're using the min and max
            coordinate of that function.
    Args:
    Returns:
    Raises:
    '''
    return stable_z_on_aabb(body, get_aabb(surface, link=surface_link))

def stable_z_on_aabb(body, aabb):
    center, extent = get_center_extent(body)
    _, upper = aabb
    return (upper + extent/2 + (get_xyz_point(body) - center))[2]

def can_collide(body, link=BASE_LINK):
    '''
    Checks if a body has physics collision logic.
    Args:
    Returns:
    Raises:
    '''
    collision_data = [CollisionShapeData(*tup) for tup in \
                      p.getCollisionShapeData(body, link, physicsClientId=CLIENT)]

    return len(collision_data) != 0

def aabb_union(aabbs):
    '''
    TODO
    '''
    if not aabbs:
        return None
    if len(aabbs) == 1:
        return aabbs[0]

    d = len(aabbs[0][0])
    lower = [min(aabb[0][k] for aabb in aabbs) for k in range(d)]
    upper = [max(aabb[1][k] for aabb in aabbs) for k in range(d)]
    return AABB(lower, upper)
