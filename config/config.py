import os
# fallback to cpu if mps is not available for specific operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
import torch

# data directory
DATA_DIR = '/home/aidara/augmented_imitation_learning/training_data/'

# checkpoint directory
CHECKPOINT_DIR = 'checkpoints/'

# device
device = 'cpu'
if torch.cuda.is_available(): device = 'cuda'
print(f"Using device: {device}")
#if torch.backends.mps.is_available(): device = 'mps'
os.environ['DEVICE'] = device

# robot port names
ROBOT_PORTS = {
    'leader': '/dev/tty.usbmodem57380045221',
    'follower': '/dev/tty.usbmodem57380046991'
}


# task config (you can add new tasks)
TASK_CONFIG = {
    'dataset_dir': DATA_DIR,
    'episode_len': 500,
    'state_dim': 9,
    'action_dim': 9,
    'cam_width': 1280,
    'cam_height': 720,
    'camera_names': ['33137761', '36829049', '39725782'],
    'camera_port': 0
}


# policy config
POLICY_CONFIG = {
    'lr': 0.375e-5,
    'device': device,
    'num_queries': 25,
    'kl_weight': 10,
    'hidden_dim': 512,
    'dim_feedforward': 3200,
    'lr_backbone': 1e-5,
    'backbone': 'resnet18',
    'enc_layers': 4,
    'dec_layers': 7,
    'nheads': 8,
    'camera_names': ['33137761', '36829049', '39725782'],
    'policy_class': 'ACT',
    'temporal_agg': True
}

# training config
TRAIN_CONFIG = {
    'seed': 42,
    'num_epochs': 8000,
    'batch_size_val': 3,
    'batch_size_train': 3,
    'eval_ckpt_name': 'policy_epoch_4600_seed_42.ckpt',
    'checkpoint_dir': CHECKPOINT_DIR
}