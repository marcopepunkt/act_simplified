from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG, ROBOT_PORTS  # must import first
from datetime import datetime
import os
import cv2
import torch
import threading
import signal
import numpy as np
import time
import argparse
import pickle

import pyzed.sl as sl

from franka_py import Robot, Controller, move_to_joint_position, Gripper

from training.utils import *

# Global variables
zed_dict = {}
img_dict = {}
timestamp_dict = {}
thread_dict = {}
cfg = {}

# parse the task name via command line
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='Cube_in_box')
args = parser.parse_args()
task = args.task

# config
cfg = TASK_CONFIG
policy_config = POLICY_CONFIG
train_cfg = TRAIN_CONFIG
device = os.environ['DEVICE']


zed_dict = {}
img_dict = {}
timestamp_dict = {}
thread_dict = {}
stop_signal = False


def grab_run(id):
    global stop_signal
    global zed_dict
    global timestamp_dict
    global img_dict
    #global depth_list

    runtime = sl.RuntimeParameters()
    while not stop_signal:
        err = zed_dict[id].grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            # Create a temporary Mat to hold the BGR image
            #temp_img = sl.Mat()
            zed_dict[id].retrieve_image(img_dict[id], sl.VIEW.LEFT)
            # Convert BGR to RGB using OpenCV
            # img_array = temp_img.get_data()
            # img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            # # Store the RGB image in img_dict
            # img_dict[id].set_data(img_rgb)
            timestamp_dict[id] = zed_dict[id].get_timestamp(sl.TIME_REFERENCE.CURRENT).data_ns
        time.sleep(0.01) #1ms
    zed_dict[id].close()
	

def signal_handler(signal, frame):
    global stop_signal
    stop_signal=True
    time.sleep(0.5)
    exit()


def init_zed_cameras(camera_names):
    """
    Initialize multiple ZED cameras
    
    Args:
        camera_names: List of camera names/identifiers
        
    Returns:
        Dictionary of initialized ZED camera objects
    """
    global stop_signal
    global zed_dict
    global img_dict
    global timestamp_dict
    global thread_dict
    signal.signal(signal.SIGINT, signal_handler)

    print("Running...")
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 15  # The framerate is lowered to avoid any USB3 bandwidth issues

    
    cameras = sl.Camera.get_device_list()
    available_serial_numbers = [cam.serial_number for cam in cameras]
    assert set(available_serial_numbers) == set(int(cn) for cn in camera_names), f"Camera names do not match: {available_serial_numbers} != {camera_names}"
    
    #List and open cameras
   
    index = 0
    for cam in available_serial_numbers:
        init.set_from_serial_number(cam)
        zed_dict[cam] = sl.Camera()
        img_dict[cam] = sl.Mat()
        #depth_list.append(sl.Mat())
        timestamp_dict[cam] = 0
        status = zed_dict[cam].open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            zed_dict[cam].close()

    #Start camera threads
    for cam_id, cam in zed_dict.items():
        if cam.is_opened():
            thread_dict[cam_id] = threading.Thread(name=f"ZED Grabber {cam_id}", target=grab_run, args=(cam_id,))
            thread_dict[cam_id].start()
    
    print("Started Thread grabber")
    
    return img_dict


if __name__ == "__main__":
    # Initialize all ZED cameras
    img_dict = init_zed_cameras(cfg['camera_names'])
    print("Initialized ZED cameras")
    if not zed_dict:
        print("No ZED cameras available. Exiting...")
        exit(1)
    # init follower
    follower = Robot("192.168.1.200")
    gripper = Gripper("192.168.1.200")
    #gripper.homing()
    print("Initialized Robot")

    project_dir = args.task
    # load the policyf
    ckpt_path = os.path.join(train_cfg['checkpoint_dir'], project_dir, train_cfg['eval_ckpt_name'])
    policy = make_policy(policy_config['policy_class'], policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
    print(loading_status)
    policy.to(device)
    policy.eval()

    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(train_cfg['checkpoint_dir'], project_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    
    # This is a try. 
    #post_process = lambda a: a * stats['qpos_mean'] + stats['qpos_mean']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    query_frequency = policy_config['num_queries']
    if policy_config['temporal_agg']:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    # bring the follower to the leader
    # Warm up all cameras
    state = follower.read_once()
    gripper_state = gripper.read_once()
    gripper_threshhold = 0.035
    
    kp = 0.5* np.array([100, 100, 100, 100, 100, 100, 50])
    controller = Controller(follower, state.q,kp)
    
    #home_joint_positions = np.array([0, -np.pi/4, 0, -3 * np.pi/4, 0, np.pi/2, np.pi/4])
    controller.start()
    print("Controller started")
    # home_position = np.array([-0.46617755,  0.29277581,  0.55141121, -2.38110638, -0.1030409,   3.955971, -2.46293664])
    # controller.update_target(home_position)
    # time.sleep(5) # wait for the robot to reach the home positcion
    print("Robot reached home position")
    joint_angles, joint_velocities = controller.get_current_state()
    
    
    # Get initial observation from all cameras
    obs = {
        'qpos': np.concatenate([joint_angles, [gripper_state.width, gripper_state.width]], axis=0),
        'qvel': np.concatenate([joint_velocities, [0, 0]], axis=0),
        'images': {int(cn): cv2.cvtColor(img_dict[int(cn)].get_data(), cv2.COLOR_BGR2RGB) for cn in cfg['camera_names']}
    }
    
    print("Starting :)")
    n_rollouts = 1
    for i in range(n_rollouts):
        ### evaluation loop
        if policy_config['temporal_agg']:
            all_time_actions = torch.zeros([cfg['episode_len'], cfg['episode_len']+num_queries, cfg['state_dim']]).to(device)
        qpos_history = torch.zeros((1, cfg['episode_len'], cfg['state_dim'])).to(device)
        with torch.inference_mode():
            print("Evaluation loop")
             # init buffers
            obs_replay = []
            action_replay = []
            
            print("Controller started")
            
            for t in range(cfg['episode_len']):
                
                # allow aborting the loop
                k = cv2.waitKey(1)
                if k == ord(' '):
                    print("Aborting evaluation loop")
                    break
                
                print(f"Step {t}")
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = np.asarray(qpos)
                qpos = np.ascontiguousarray(qpos).astype('float')
                qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(obs['images'], cfg['camera_names'], device)

                if t % query_frequency == 0:
                    all_actions = policy(qpos, curr_image)
                    print("_______________________Did inference")
                if policy_config['temporal_agg']:
                    print("Temporal aggregation enabled")
                    all_time_actions[[t], t:t+num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights.astype(np.float32)).to(device).unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, t % query_frequency]

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                
                ### Check if the targets are within joint limits and clip them
                low = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]) 
                high = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
                safety_offset = 1e-2
                if np.any(np.isclose(action[:7], low, atol=safety_offset)) or np.any(np.isclose(action[:7], high, atol=safety_offset)):
                    print("Warning: action is close to joint limit")
                action[:7] = np.clip(action[:7], low + safety_offset, high - safety_offset)

                #action = pos2pwm(action).astype(int)
                ### take action
                gripper_state = gripper.read_once()
                print(f"Gripper state: {gripper_state.width}")
                
                joint_angles, joint_velocities = controller.get_current_state()
                
                
                controller.update_target(action[:7])
                try:
                    if action[7] < gripper_threshhold and gripper_state.width/2 > gripper_threshhold:
                        print("Closing gripper")
                        gripper.grasp(0.0,0.5,70,0.5,0.5)
                        
                    elif action[7] > gripper_threshhold and gripper_state.width/2 < gripper_threshhold:
                        print("Opening gripper")
                        gripper.move(0.08,1)
                        
                    if gripper_state.width < 0.02:
                        print("Gripper Missed")
                        gripper.move(0.08,1)
                except Exception as e:
                    print(e)
                    print("Reconnecting to gripper")
                    del gripper
                    time.sleep(5)
                    gripper = Gripper("192.168.1.200")
                    gripper.move(0.08, 1)
                    gripper_state = gripper.read_once()
                
                ### update obs
                #time.sleep(1)
                print("Updated obs")
                # Update observation with all ZED camera images
                obs = {
                    'qpos': np.concatenate([joint_angles, [gripper_state.width/2 + 0.005, gripper_state.width/2 + 0.005]], axis=0), # Offset just temporary
                    'qvel': np.concatenate([joint_velocities, [0, 0]], axis=0),
                    'images': {int(cn): cv2.cvtColor(img_dict[int(cn)].get_data()[..., :3], cv2.COLOR_BGR2RGB) for cn in cfg['camera_names']}
                }
                ### store data
                obs_replay.append(obs)
                action_replay.append(action)
                print(action)
                print(f"Step {t}/{cfg['episode_len']}")
                time.sleep(5/30)

        print("Episode finished")
        # create a dictionary to store the data
        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        # there may be more than one camera
        for cam_name in cfg['camera_names']:
                data_dict[f'/observations/images/{cam_name}'] = []

        # store the observations and actions
        for o, a in zip(obs_replay, action_replay):
            data_dict['/observations/qpos'].append(o['qpos'])
            data_dict['/observations/qvel'].append(o['qvel'])
            data_dict['/action'].append(a)
            # store the images
            for cam_name in cfg['camera_names']:
                data_dict[f'/observations/images/{str(cam_name)}'].append(o['images'][int(cam_name)])

        #t0 = time()
        max_timesteps = len(data_dict['/observations/qpos'])
        # create data dir if it doesn't exist
        # data_dir = os.path.join(cfg["checkpoints"], project_dir)  
        # if not os.path.exists(data_dir): os.makedirs(data_dir)
        # # count number of files in the directory
        # idx = len([name for name in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, name))])
        # dataset_path = os.path.join(data_dir, f'episode_{idx}')
        # save the data
        filename = f"replay_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.hdf5"
        savepath = os.path.join("data", project_dir, filename)
        print(f"Attempting to save replay to: {savepath}")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(savepath), exist_ok=True)

        with h5py.File(savepath, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in cfg['camera_names']:
                _ = image.create_dataset(cam_name, (max_timesteps, cfg['cam_height'], cfg['cam_width'], 3), dtype='uint8',
                                        chunks=(1, cfg['cam_height'], cfg['cam_width'], 3), )
            qpos = obs.create_dataset('qpos', (max_timesteps, cfg['state_dim']))
            qvel = obs.create_dataset('qvel', (max_timesteps, cfg['state_dim']))
            # image = obs.create_dataset("image", (max_timesteps, 240, 320, 3), dtype='uint8', chunks=(1, 240, 320, 3))
            action = root.create_dataset('action', (max_timesteps, cfg['action_dim']))
            
            for name, array in data_dict.items():
                root[name][...] = array

        print(f"Saved to {savepath}")
    
    controller.stop()
    print("Evaluation finished")