import numpy as np
import time
from franka_py import Controller, Robot, move_to_joint_position


if __name__ == '__main__':
    robot = Robot("192.168.1.200")
    
    
    # state = robot.read_once()
    # kp = np.array([300, 200, 300, 200, 200, 200, 100])
    # controller = Controller(robot, state.q,kp)
    # controller.start()
    
    home_joint_positions = np.array([0, -np.pi/4, 0, -3 * np.pi/4, 0, np.pi/2, np.pi/4])
    # #home_joint_positions = np.array([ 0.6255863308906555, 0.7028061747550964, -0.6193988919258118, -2.184051275253296, 0.17927229404449463, 4.2207536697387695, -2.308631181716919])
    # controller.update_target(home_joint_positions)
    # time.sleep(5)
    # controller.stop()
    
    move_to_joint_position(robot, home_joint_positions, 1)
    