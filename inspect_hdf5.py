import h5py
import os

def inspect_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        return

    with h5py.File(dataset_path, 'r') as root:
        print("HDF5 file structure:")
        
        def print_attrs(name, obj):
            print(f"Path: {name}")
            if isinstance(obj, h5py.Dataset):
                print(f"  Shape: {obj.shape}")
                print(f"  Type: {obj.dtype}")
                if len(obj) > 0:
                    try:
                        print(f"  First element shape: {obj[0].shape if hasattr(obj[0], 'shape') else 'scalar'}")
                    except:
                        pass
            return None
        
        root.visititems(print_attrs)

if __name__ == "__main__":
    data_file = '/home/aidara/augmented_imitation_learning/training_data/randomstuff/threecolor_movement_1.hdf5'
    inspect_hdf5(dataset_path=data_file)
