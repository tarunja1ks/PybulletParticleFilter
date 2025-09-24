import cv2
import math
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import time

try:
#     # I'll install open3d later lol - Muhammad
    import open3d as o3d
except:
    pass

class Sensor():
    """
    Implements a reference/base interface
    for all other sensors. This class acts
    as the parent (base class) to other sensors.

    Every sensor needs to implement `setup()`
    and `retrieve_data()`.
    """
    def setup(self) -> None:
        """
        Setup the sensor.
        """
        raise NotImplementedError

    def retrieve_data(self) -> None:
        """
        Retrieve whatever data the sensor supposed
        to get.
        """
        raise NotImplementedError

class Camera(Sensor):
    """
    Class to manage pybullet camera.
    """

    def __init__(self, 
            car_id: int,
            configs: dict
        ) -> None:
        """
        Initialize Camera

        """
        self.POS = configs['camera_pos'] 
        self.INIT_POS = configs['camera_pos']
        self.ORI = configs['camera_ori'] if len(configs['camera_ori']) == 4 else p.getQuaternionFromEuler(configs['camera_ori'])
        self.INIT_ORI = configs['camera_ori']
        self.FOV = configs['camera_fov']
        self.ASPECT = configs['camera_aspect']
        self.NEAR = configs['camera_near']
        self.FAR = configs['camera_far']
        self.WIDTH = configs['camera_width']
        self.HEIGHT = configs['camera_height']
        self.RENDER_MODE = p.ER_BULLET_HARDWARE_OPENGL # renders images within Pybullet
        self.FLAGS = 1
        self.FX = configs['camera_fx']
        self.FY = configs['camera_fy']
        self.DIST = configs['camera_dist']
        self.TARGET_POS = [2*configs['camera_pos'][0],2*configs['camera_pos'][1],2*configs['camera_pos'][2]]
        self.UP_AXIS_INDEX = configs['camera_up_axis_index']
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)

    def setup(self) -> None:
        """
        Create a camera in the simulation.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """
        self.CAMERA = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.05, 0.02], rgbaColor=[1, 0, 0, 0.7])
        self.CAMERA = p.createMultiBody(0,-1, self.CAMERA, basePosition=self.POS, baseOrientation=self.ORI)

    def retrieve_data(self, common=True, car_state=None) -> tuple:
        """
        Create an image of the camera.

        Args:
            car_state: The car's current state. Some sensors
                       need this, some don't. It is included
                       for consistency of the API.
            common: For API consistency, same as `car_state`.
        Returns:
            tuple: RGB image, depth image, segmentation image.
        Raises:
            None.
        """

        visual_shapes = p.getVisualShapeData(-1)
        for i, shape in enumerate(visual_shapes):
            obj_id = shape[0]
            p.changeVisualShape(obj_id, -1, rgbaColor=[1,0,0,1], flags=p.VISUAL_SHAPE_BULLET_HAS_COLLISION_SHAPE, visualId = i+1)

        # Get the camera image, depth and segmentation image of the current environment
        self.IMAGE = p.getCameraImage(
            width=self.WIDTH,
            height=self.HEIGHT,
            viewMatrix= p.computeViewMatrix(
                cameraEyePosition=self.POS,
                cameraTargetPosition=self.TARGET_POS,
                cameraUpVector=self.UP_AXIS_INDEX
            ),
            projectionMatrix=p.computeProjectionMatrixFOV(
                fov=self.FOV,
                aspect=self.ASPECT,
                nearVal=self.NEAR,
                farVal=self.FAR
            ),
            shadow=1,
            renderer=self.RENDER_MODE,
            flags=self.FLAGS

            
        )
        self.POINT_CLOUD = None
        # Get point cloud from the depth image
        self.create_point_cloud()

        return self.IMAGE, self.POINT_CLOUD
    
    def get_camera_image_rgb(self) -> None:
        """
        Get the RGB image of the camera.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """
        return self.IMAGE[2]
    
    def get_camera_image_depth(self) -> None:
        """
        Get the depth image of the camera.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """
        return self.IMAGE[3]
    
    def get_camera_image_segmentation(self) -> None:
        """
        Get the segmentation image of the camera.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """
        return self.IMAGE[4]
        
    def create_point_cloud(self) -> None:
        """
        Create a point cloud from the depth image of the camera.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """

        self.reconstructor = Reconstruction([self.WIDTH,self.HEIGHT],[self.FX,self.FY],[self.WIDTH/2,self.HEIGHT/2],[[1,0,0],[0,1,0],[0,0,1]],[0,0,0])
        
        image = self.get_camera_image_rgb()
        depth = self.get_camera_image_depth()

        # Sample the image and depth map to only take every nth pixel
        n = 5
        image = image[::n,::n,:]
        depth = depth[::n,::n]

        # Use open3d to make an RGBD image and a point cloud from the depth map and the image
        depth = np.dstack((depth,depth,depth))
        image = o3d.geometry.Image(image)
        depth = o3d.geometry.Image(depth)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image, depth, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

        self.POINT_CLOUD = pcd

        # # Using original code to make point cloud
        # self.reconstructor.add_image(image,depth)
        # self.reconstructor.point_cloud()
        # self.reconstructor.visualize()
        # self.POINT_CLOUD = self.reconstructor.pcd_o3d

    def get_point_cloud(self) -> None:
        """
        Get the point cloud of the camera.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """
        return self.POINT_CLOUD
    
    def display_point_cloud(self) -> None:
        """
        Display the point cloud of the camera.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """
        o3d.visualization.draw_geometries([self.POINT_CLOUD])

    def update_camera_orientation(self, rotation: list) -> None:
        """
        Rotate the camera.

        Args:
            rotation: Rotation.
        Returns:
            None.
        Raises:
            None.
        """

        self.ORI = p.getQuaternionFromEuler(rotation) if len(rotation) == 3 else rotation

        # Rotate the camera in the pybullet simulation.
        p.resetBasePositionAndOrientation(self.CAMERA, self.POS, self.ORI)

        print("Camera orientation:", self.ORI)
        # print orientation in euler angles
        print("Camera orientation (euler):", p.getEulerFromQuaternion(self.ORI))

    def update_camera_position(self, vector: list) -> None:
        """
        Translate the camera.

        Args:
            vector: Position of the camera.
        Returns:
            None.
        Raises:
            None.
        """

        self.POS = vector

        # Translate the camera in the pybullet simulation.
        p.resetBasePositionAndOrientation(self.CAMERA, self.POS, self.ORI)
        
        euler = p.getEulerFromQuaternion(self.ORI)
        magnitude_init_pos = np.sqrt(self.INIT_POS[0]**2 + self.INIT_POS[1]**2)
        self.TARGET_POS = [self.POS[0] + magnitude_init_pos*np.cos(euler[2] * np.pi / 180), self.POS[1] + magnitude_init_pos * np.sin(euler[2] * np.pi/180), self.INIT_POS[2]]
        
        print("Camera position:", self.POS)

class Reconstruction:

    """
    Class to manage 3D reconstruction of PCD from depth map
    
    """

    def __init__(self,img_size,focal_length,img_center,rotation,translation):
        self.rotation = np.array(rotation)
        self.translation = np.array(translation)
        self.img_id = ""         
        self.fx = focal_length[0]
        self.fy = focal_length[1]
        self.cx = img_center[0]
        self.cy = img_center[1]

    def add_image(self,image,depth_map):
        self.image = image
        self.depth_map = depth_map
        
    def display(self): # Display the image and the depth map; does not need to be called
        fig = plt.figure()
        fig.add_subplot(1,2,1)
        plt.imshow(cv2.cvtColor(self.image,cv2.COLOR_BGR2RGB))
        plt.title("Image")
        fig.add_subplot(1,2,2)
        plt.imshow(self.depth_map)
        plt.title("Depth Map")
        plt.show()          

    def point_cloud(self):
        start = time.time()
        self.pcd = np.hstack(
            (np.transpose(np.nonzero(self.depth_map)), np.reshape(self.depth_map[np.nonzero(self.depth_map)], (-1,1)) )
        )  # (xxx, 3)

        self.pcd[:, [0, 1]] = self.pcd[:, [1, 0]]  # swap x and y axis since they are reversed in image coordinates

        self.pcd[:, 0] = (self.pcd[:, 0] - self.cx) * self.pcd[:, 2] / self.fx
        self.pcd[:, 1] = (self.pcd[:, 1] - self.cy) * self.pcd[:, 2] / self.fy

        self.colors = np.flip(self.image[np.nonzero(self.depth_map)], axis=1)

    def translate_point_cloud(self,vector):
        self.pcd += vector

    def rotate_point_cloud(self,rotate):    
        self.pcd = np.matmul(rotate,self.pcd.T).T

    def visualize(self):
         self.pcd_o3d = o3d.geometry.PointCloud()
         self.pcd_o3d.points = o3d.utility.Vector3dVector(self.pcd)
         # self.pcd_o3d.colors = o3d.utility.Vector3dVector(self.colors/255)

class Lidar(Sensor):
    """
    Lidar class to manage lidar module in PyBullet.
    """
    def __init__(self,
            car_id: int,
            configs: dict
        ) -> None:
        """
        Initialize the lidar. 

        Note that the specs of the lidar are specified in 
        the car's config file, from which the car will 
        call a Lidar instance and pass these lidar configs.

        Args:
            car_id: The unique `car_id` passed by Car object.
            configs:
                - lidar_joints: 
                - lidar_angle1: 
                - lidar_angle2: 
                - num_rays:
                - ray_start_len:
                - ray_len:
                - hit_color:
                - miss_color:
        Returns:
            None.
        Raises:
            None.
        """
        assert isinstance(configs, dict)

        self.LIDAR_JOINTS   = configs['lidar_joints']
        self.START_ANGLE    = configs['lidar_angle1']
        self.END_ANGLE      = configs['lidar_angle2']
        self.NUM_RAYS       = configs['num_rays']
        self.RAY_START_LEN  = configs['ray_start_len']
        self.RAY_LEN        = configs['ray_len']
        self.HIT_COLOR      = configs['hit_color']
        self.MISS_COLOR     = configs['miss_color']

        # Get the car's id.
        self.car_id = car_id

    def setup(self) -> None:
        """
        Setups Lidar object (a one-time setup).

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """
        a = self.START_ANGLE*(math.pi/180)
        b = (self.END_ANGLE - self.START_ANGLE)*(math.pi/180)

        ray_from, ray_to = [], []
        for i in range(self.NUM_RAYS):
            theta = float(a) + (float(b) * (float(i)/self.NUM_RAYS))
            x1 = self.RAY_START_LEN*math.sin(theta)
            y1 = self.RAY_START_LEN*math.cos(theta)
            z1 = 0

            x2 = self.RAY_LEN*math.sin(theta)
            y2 = self.RAY_LEN*math.cos(theta)
            z2 = 0

            ray_from.append([x1, y1, z1])
            ray_to.append([x2, y2, z2])

        self.ray_from = ray_from
        self.ray_to = ray_to

        ray_ids = []
        for i in range(self.NUM_RAYS):
            ray_ids.append(
                    p.addUserDebugLine(
                            self.ray_from[i],
                            self.ray_to[i],
                            self.MISS_COLOR,
                            parentObjectUniqueId=self.car_id,
                            parentLinkIndex=self.LIDAR_JOINTS
                        )
                    )
        self.ray_ids = ray_ids

    def retrieve_data(self, common=True, car_state=None) -> tuple:
        """
        In most mobile robot applicatoins, each ray (or lidar)
        gives two information: 1. range, 2. bearing (w.r.t car's 
        yaw). Range is how far the ray reached - if it hits something
        than it should be less than it's max length. Bearing tells
        us where this scan occurs in the map w.r.t the car's frame.

        However, PyBullet has nicely given us the coordinates in
        the world frame of obstacles that a ray hits.

        Hence, this function will implement both cases. Set
        common=False for the second case (since the first case
        is more common in mobile robots problems)

        Args:
            car_state: The car's current state (x, y, yaw). Some sensors
                       need this, some don't. It is included
                       for consistency of the API.
            common: For API consistency, same as `car_state`.
        Returns:
            rays_data: A numpy array needed to simulate LiDAR.
            coords: Coordinate (x, y) of hit points in world
                    coordinate. For a ray that does not hit an object,
                    the value is None.
            dists: Distance to hit object for every ray. For a ray that
                   does not hit an object, the value is self.RAY_LEN

        """
        num_threads = 0
        rays_data = p.rayTestBatch(
                    self.ray_from, 
                    self.ray_to, 
                    num_threads, 
                    parentObjectUniqueId=self.car_id, 
                    parentLinkIndex=self.LIDAR_JOINTS
                )

        # Convert `results` to numpy array to leverage Numpy's speed.
        rays_data = np.array(rays_data, dtype=object)

        # Get the rays which does not hit an object
        no_hit = np.where(rays_data[:, 2] == 1.)[0]

        # Convert `coords` to numpy array to leverage Numpy's speed.
        # `coords` will be of shape (NUM_RAYS, 3) and coords[i] contains 
        # [x, y, z] coordinates in world coordinate, for each rays.
        coords = np.array(rays_data[:, 3], dtype=object) # shape: (NUM_RAYS, )
        coords = np.stack(coords).astype(np.float32)     # shape: (NUM_RAYS, 3)
        coords = coords[:, :2] # We only need (x, y) coords.

        # Get the distances
        x, y, yaw = car_state
        dists = np.sqrt((coords[:, 0] - x)**2 + (coords[:, 1] - y)**2) # (NUM_RAYS, )

        # Set distances to self.RAY_LEN for rays that doesn't hit an object.
        dists[no_hit]  = self.RAY_LEN

        # Get bearing data
        angle = np.arctan2(coords[:, 1] - y, coords[:, 0] - x)
        angle *= 180.0/np.pi
        bearings = angle - yaw
        bearings = np.where(bearings < 0, bearings + 360, bearings)

        if common:
            return rays_data, dists, bearings

        # set coord to None for rays with no hit.
        coords[no_hit] = None # Set coord to None for rays with not hit.
        return rays_data, dists, coords

    def simulate(self, rays_data: np.ndarray) -> None:
        """
        Simulate the rays from lidar. User should
        not call this function when running/testing
        algorithm to save compute power, but good 
        to use for visualization and debugging.

        This method will be called iff 
        the `SIMULATE_LIDAR` boolean is set
        to True and LIDAR fps is satisfied (see `main.py`).

        Args:
            rays_data: TODO (refer p.rayTestBatch return values)
        Returns:
        Raises:
        """
        for i in range (self.NUM_RAYS):
            hit_object_id = rays_data[i][0]
            hit_fraction  = rays_data[i][2]
            hit_position  = rays_data[i][3]

            if (hit_fraction==1.):
                # No object hit.
                p.addUserDebugLine(
                            self.ray_from[i], 
                            self.ray_to[i], 
                            self.MISS_COLOR, 
                            replaceItemUniqueId=self.ray_ids[i], 
                            parentObjectUniqueId=self.car_id, 
                            parentLinkIndex=self.LIDAR_JOINTS
                        )
            else:
                # Object hit.
                localHitTo = [self.ray_from[i][0]+hit_fraction*(self.ray_to[i][0]-self.ray_from[i][0]), \
                              self.ray_from[i][1]+hit_fraction*(self.ray_to[i][1]-self.ray_from[i][1]), \
                              self.ray_from[i][2]+hit_fraction*(self.ray_to[i][2]-self.ray_from[i][2])]
                p.addUserDebugLine(
                            self.ray_from[i], 
                            localHitTo, 
                            self.HIT_COLOR, 
                            replaceItemUniqueId=self.ray_ids[i], 
                            parentObjectUniqueId=self.car_id, 
                            parentLinkIndex=self.LIDAR_JOINTS
                        )
