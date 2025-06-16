import os
import h5py
import numpy as np
import time

from franka_py import Robot, PDController, move_to_joint_position, Gripper, Controller


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        qpos = root['/observations/qpos'][()]
        # qvel = root['/observations/qvel'][()]
        # image_dict = dict()
        # for cam_name in root[f'/observations/images/'].keys():
        #     #image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]
        # action = root['/action'][()]

    return qpos#, #qvel, action, image_dict



def main():
    # play cam video
    data_file = '/home/aidara/augmented_imitation_learning/training_data/Cube_in_box2/episode_2.hdf5'
    #data_file = 'data/demo/trained.hdf5'
    qpos = load_hdf5(dataset_path=data_file)


    robot = Robot("192.168.1.200")
    #set_default_behavior(robot)

    # Initialize gripper
    print("Connecting to gripper...")
    gripper = Gripper("192.168.1.200")  # Same IP as robot
    print("Connected to gripper")
    gripper.homing()
    
    frame_counter = 1
    gripper_threshold = 0.039  # threshold for gripper action

    # Perform homing for the gripper
   
    state = robot.read_once()
    home = np.array([0, -np.pi/4, 0, -3 * np.pi/4, 0, np.pi/2, np.pi/4])
    move_to_joint_position(robot, home.tolist(),0.5 )
    
    # time.sleep(0.5)
    # move_to_joint_position(robot, qpos[0][:7].tolist(),0.5 )
    # print("moved to qpos")
    # time.sleep(0.5)
    
    # #move_to_joint_position(robot, [ 0.61059354,  0.84743161, -0.4139109,  -2.9718,     -1.07808982,  3.6525, -0.86189719 ],0.5 )
    # print("moved to home")
    # time.sleep(0.5)
    # print("moveds")
    kp = 0.5 * np.array([100, 100, 100, 100, 100, 100, 50])
    pd_controller = Controller(robot, state.q,kp)
    pd_controller.start()
    
    pd_controller.update_target(qpos[0][:7].tolist())
    
    #pd_controller.update_target( np.array([0.4937153, 0.78781367, -0.4497713, -2.9718, -0.70264357, 3.6525, -1.26562544]))
    time.sleep(10)
    # for i,q in enumerate(qpos):
    #     if i %30 != 0:
    #         print("skipping")
    #         continue
    #     # Update robot arm position
    #     joint_state = q[:7]
    #     joint_state = np.array(joint_state)
    #     print(f"acquired {joint_state}")
    #     pd_controller.update_target(joint_state)
        
    #     # Update gripper position based on threshold
    #     gripper_width = q[7]
        
    #     # Track consecutive frames above/below threshold
    #     if not hasattr(main, 'threshold_counter_open'):
    #         main.threshold_counter_open = 0
    #         main.threshold_counter_close = 0
    #         main.last_gripper_state = "open"
        
    #     if gripper_width > gripper_threshold:
    #         main.threshold_counter_open += 1   
    #         main.threshold_counter_close = 0
    #     else:
    #         main.threshold_counter_close += 1
    #         main.threshold_counter_open = 0
            
    #     # Change gripper state after 3 consecutive frames in either direction
    #     if main.threshold_counter_open >= frame_counter and main.last_gripper_state != "open":
    #         gripper.move(0.08,1)
    #         main.last_gripper_state = "open"
    #         print("Opening gripper")
    #     elif main.threshold_counter_close >= frame_counter and main.last_gripper_state != "closed":
    #         gripper.grasp(0.0,0.5,70,0.5,0.5)
    #         main.last_gripper_state = "closed"
    #         print("Closing gripper")
            
    #     # if main.last_gripper_state == "closed":
    #     #     gripper.grasp(0.02,1,70,0.01,0.01)
            
    #     #gripper_success = gripper.move(gripper_width*2, gripper_speed)
        
    #     time.sleep(1)   #move_to_joint_position(robot,q.tolist(), 0.5)
    #     print("updated target")
    #     print(f"STEP {i} successfully executed")
    print("Done")
    pd_controller.stop()

if __name__ == "__main__":
    # Run the main function in an asyncio event loop
    main()
    #main()    