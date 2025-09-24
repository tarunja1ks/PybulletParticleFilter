import math
import pybulletPF as p
from pynput import keyboard

UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'
Y = 'y'
H = 'h'
J = 'j'
K = 'k'
U = 'u'
B = 'b'
N = 'n'
M = 'm'

numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

class PID():
    def __init__(self, path, kp=0.3, ki=0.1, kd=0.01, debug=False) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.previous_error = 0
        self.integral_term = 0
        self.derivative_term = 0

        self.path = path
        self.target_index = 0

        if debug:
            self.__add_sliders()

    def navigate(self, x, y, yaw):
        curr_target = self.path[self.target_index]
        dx = curr_target[0] - x
        dy = curr_target[1] - y
        distance = math.sqrt(dx ** 2 + dy ** 2)
        
        if distance < 0.5:
            self.target_index += 1
            if self.target_index >= len(self.path):
                return 0, 0  # Stop if last target is reached

        cte = dy * math.cos(yaw) - dx * math.sin(yaw)
        self.integral_term += cte
        self.derivative_term = cte - self.previous_error
        self.previous_error = cte

        velocity = (self.kp * cte) + (self.ki * self.integral_term) + (self.kd * self.derivative_term) + 10
        steering = math.atan2(-cte, distance)

        velocity = max(-50, min(50, velocity))
        steering = max(-1, min(1, steering))

        print(f"Velocity: {velocity}, Steering: {steering}")

        return velocity, steering
    
    def update_ks(self):
        """
        For tuning and debugging.
        """
        self.kp = p.readUserDebugParameter(self.kp_slider)
        self.ki = p.readUserDebugParameter(self.ki_slider)
        self.kd = p.readUserDebugParameter(self.kd_slider)
    
    def __add_sliders(self) -> None:
        """
        Add sliders on the sim env.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """
        self.kp_slider = p.addUserDebugParameter(
                    "Kp",
                    0, 10, self.kp
                )
        self.ki_slider = p.addUserDebugParameter(
                    "Ki",
                    0, 0.1, self.ki
                )
        self.kd_slider = p.addUserDebugParameter(
                    "Kd",
                    0, 1, self.kd
                )

class Stanley():
    def __init__(
            self,
            path,
            k=0.1,
            previous_error=0,
            target_index=0,
            debug=False
        ) -> None:

        self.k = k

        self.previous_error = previous_error
        self.path = path
        self.target_index = target_index

        if debug:
            pass

    def navigate(self, x, y, yaw):
        curr_x, curr_y, curr_yaw = x, y, yaw

        curr_target = self.path[self.target_index]
        while True:
            dx = curr_target[0] - curr_x
            dy = curr_target[1] - curr_y
            distance = math.sqrt(dx ** 2 + dy ** 2)
            if distance < 0.5:
                # within 0.5 meters of target point, switch to next target point
                if self.target_index < len(self.path) - 1:
                    self.target_index += 1
                    curr_target = self.path[self.target_index]
                else:  # reached final target point
                    break
            else:
                break

        # Calculate cross track error (cte).
        cte = dy * math.cos(curr_yaw) - dx * math.sin(curr_yaw)

        # Calculate the heading error (theta_e)
        desired_yaw = math.atan2(curr_target[1] - curr_y, curr_target[0] - curr_x)
        theta_e = desired_yaw - curr_yaw
        theta_e = math.atan2(math.sin(theta_e), math.cos(theta_e))

        # Compute velocity command (constant velocity in this case)
        velocity = 13

        # Calculate the steering angle using the Stanley controller equation
        steering = -(theta_e + math.atan(self.k * cte / (velocity + 0.001)))

        # limit steering angle to range [-1, 1]
        steering = max(-1.0, min(1.0, steering))

        return velocity, steering

class PurePursuit():
    def __init__(
            self,
            path,
            look_ahead_distance=0.5,
            max_steering_angle=1.0,
            debug=False
        ) -> None:

        self.look_ahead_distance = look_ahead_distance
        self.max_steering_angle = max_steering_angle

        self.path = path
        self.target_index = 0

        if debug:
            pass

    def navigate(self, x, y, yaw):
        curr_x, curr_y, curr_yaw = x, y, yaw

        curr_target = self.path[self.target_index]
        while True:
            dx = curr_target[0] - curr_x
            dy = curr_target[1] - curr_y
            distance = math.sqrt(dx ** 2 + dy ** 2)
            if distance < 0.5:
                # within 0.5 meters of target point, switch to next target point
                if self.target_index < len(self.path) - 1:
                    self.target_index += 1
                    curr_target = self.path[self.target_index]
                else:  # reached final target point
                    break
            else:
                break

        # Calculate target point based on look ahead distance
        while self.target_index < len(self.path) - 1:
            next_target = self.path[self.target_index + 1]
            dx = next_target[0] - curr_x
            dy = next_target[1] - curr_y
            target_distance = math.sqrt(dx ** 2 + dy ** 2)
            if target_distance > self.look_ahead_distance:
                break
            else:
                self.target_index += 1
                curr_target = next_target

        # Calculate steering angle
        dx = curr_target[0] - curr_x
        dy = curr_target[1] - curr_y
        alpha = math.atan2(dy, dx) - curr_yaw
        steering = -math.atan2(2.0 * 0.5 * math.sin(alpha), self.look_ahead_distance)

        # Limit steering angle to range [-max_steering_angle, max_steering_angle]
        steering = max(-self.max_steering_angle, min(self.max_steering_angle, steering))

        # Compute velocity
        velocity = 13.0  # or whatever velocity you want to maintain

        return velocity, steering


class KeyboardController():

    UP, DOWN, LEFT, RIGHT = 'up', 'down', 'left', 'right'  # Define constants for directions
    H, J, K, U, N, M = 'h', 'j', 'k', 'u', 'n', 'm' # to control the robotic arm

    """
    Implements a manual keyboard controller
    using arrow keys
    """
    def __init__(
            self, 
            ctrl_fps,
            acceleration=10.0,
            max_velocity=50.0,
            steering_angle_per_sec=90,
            arm = False, # presence of robotic arm
            forward = True, # Forward or inverse kinematics
            num_joints = 7
        ):
        self.CTRL_FPS = ctrl_fps
        self.acceleration = acceleration  # 10 units per second^2
        self.max_velocity = max_velocity  # Maximum velocity of 50 units per second
        self.steering_angle_per_sec = steering_angle_per_sec  # Steering angle changes by 90 degrees per second

        self.velocity = 0
        self.steering = 0

        self.current_keys = set()
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

        self.arm = False
        if arm:
            self.forward = forward
            self.arm = True
            if self.forward:
                self.joint_angles = [0.0] * num_joints
            else:
                self.arm_delta_x = 0.0
                self.arm_delta_y = 0.0
                self.arm_z = 1.0

    def navigate(self, x, y, yaw):
        """
        x, y, yaw are added as parameters for API consistency
        eventhough this controller does no need these values.
        """
        # Calculate steering
        if RIGHT in self.current_keys:
            self.steering += self.steering_angle_per_sec / self.CTRL_FPS
            self.steering = min(self.steering, 1.0)
        elif LEFT in self.current_keys:
            self.steering -= self.steering_angle_per_sec / self.CTRL_FPS
            self.steering = max(self.steering, -1.0)
        else:
            self.steering = 0

        # Calculate velocity
        if UP in self.current_keys:
            self.velocity += self.acceleration / self.CTRL_FPS
            self.velocity = min(self.velocity, self.max_velocity)
        elif DOWN in self.current_keys:
            self.velocity -= self.acceleration / self.CTRL_FPS
            self.velocity = max(self.velocity, -self.max_velocity)
        else:
            self.velocity = 0

        if self.arm:
            if self.forward:
                if '1' in self.current_keys:
                    self.joint_angles[0] += 0.01
                elif Y in self.current_keys:
                    self.joint_angles[0] -= 0.01
                elif '2' in self.current_keys:
                    self.joint_angles[1] += 0.01
                elif U in self.current_keys:
                    self.joint_angles[1] -= 0.01
                elif '3' in self.current_keys:
                    self.joint_angles[2] += 0.01
                elif H in self.current_keys:
                    self.joint_angles[2] -= 0.01
                elif '4' in self.current_keys:
                    self.joint_angles[3] += 0.01
                elif J in self.current_keys:
                    self.joint_angles[3] -= 0.01
                elif '5' in self.current_keys:
                    self.joint_angles[4] += 0.01
                elif K in self.current_keys:
                    self.joint_angles[4] -= 0.01
                elif '6' in self.current_keys:
                    self.joint_angles[5] += 0.01
                elif N in self.current_keys:
                    self.joint_angles[5] -= 0.01
                elif '7' in self.current_keys:
                    self.joint_angles[6] += 0.01
                elif M in self.current_keys:
                    self.joint_angles[6] -= 0.01                    

                return self.velocity, self.steering, self.joint_angles
            else:
                if J in self.current_keys:
                    self.arm_z += 0.01
                elif M in self.current_keys:
                    self.arm_z -= 0.01
                elif H in self.current_keys:
                    self.arm_delta_x -= 0.01
                elif K in self.current_keys:
                    self.arm_delta_x += 0.01
                elif U in self.current_keys:
                    self.arm_delta_y += 0.01
                elif N in self.current_keys:
                    self.arm_delta_y -= 0.01
                
                target_x = x + self.arm_delta_x
                target_y = y + self.arm_delta_y
                # Control the robotic arm
                target_pos = [target_x, target_y, self.arm_z]

                return self.velocity, self.steering, target_pos
        
        return self.velocity, self.steering

    def on_press(self, key):
        try:
            if key == keyboard.Key.up:
                self.current_keys.add(UP)
            elif key == keyboard.Key.down:
                self.current_keys.add(DOWN)
            elif key == keyboard.Key.left:
                self.current_keys.add(LEFT)
            elif key == keyboard.Key.right:
                self.current_keys.add(RIGHT)
            elif key.char in [H, J, K, U, N, M, B, Y]:
                self.current_keys.add(key.char)
            elif key.char in numbers:
                self.current_keys.add(key.char)
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            if key == keyboard.Key.up:
                self.current_keys.discard(UP)
            elif key == keyboard.Key.down:
                self.current_keys.discard(DOWN)
            elif key == keyboard.Key.left:
                self.current_keys.discard(LEFT)
            elif key == keyboard.Key.right:
                self.current_keys.discard(RIGHT)
            elif key.char in [H, J, K, U, N, M, B, Y]:
                self.current_keys.discard(key.char)
            elif key.char in numbers:
                self.current_keys.discard(key.char)
        except AttributeError:
            pass

    def press_key(self, key):
        self.current_keys.add(key)

    def release_key(self, key):
        self.current_keys.discard(key)

if __name__ == "__main__":
    controller = KeyboardController(ctrl_fps=100)
    listener = keyboard.Listener(on_press=controller.on_press, on_release=controller.on_release)
    listener.start()
    while True:
        v, s = controller.navigate(0,0,0)
        print(v, s)