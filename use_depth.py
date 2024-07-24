import h5py

# This is an example of using depth data
filename = 'check/a99731685e4a44c7b3ce80e41633f486/driving+school_4k-diffuse/train/depth/000.h5'
with h5py.File(filename, 'r') as hdf:
    dataset = hdf['depth']
    data = dataset[:]
    print(data.shape)
    print(data.max())