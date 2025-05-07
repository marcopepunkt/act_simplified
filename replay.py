import os
import h5py
import mediapy as media
import matplotlib.pyplot as plt
import numpy as np
import time
import asyncio 

from franka_py import Robot, PDController, set_default_behavior, move_to_joint_position


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]
        action = root['/action'][()]

    return qpos, qvel, action, image_dict



async def main():
    # play cam video
    data_file = '/home/aidara/augmented_imitation_learning/training_data/randomstuff/threecolor_movement_1.hdf5'
    #data_file = 'data/demo/trained.hdf5'
    qpos, qvel, action, image_dict = load_hdf5(dataset_path=data_file)


    robot = Robot("192.168.1.200")
    #set_default_behavior(robot)

    state = robot.read_once()


    move_to_joint_position(robot, qpos[0][:7].tolist(), 0.5)

    
    pd_controller = PDController(robot, np.array(state.q))
    pd_controller.start()

    for i,q in enumerate(qpos):
        q = q[:7]
        q = np.array(q)
        print(f"aqquired {q}")
        pd_controller.update_target(q)
        await asyncio.sleep(1/30)    #move_to_joint_position(robot,q.tolist(), 0.5)
        print("updated target")
        print(f"STEP {i} successfully executed")
    
    pd_controller.stop()

if __name__ == "__main__":
    # Run the main function in an asyncio event loop
    asyncio.run(main())
    #main()    