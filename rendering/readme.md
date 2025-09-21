# Steps for Creating a Dataset

1. Randomly generate the dataset and output according to the format specification below.
2. Run the script to calculate overlap scores for each image pair. The overlaps
   get saved as `overlaps.pkl`.
3. Convert the dataset to HDF5 format using the `convert_to_hdf5.py` script. After converting
  to HDF5, the original files can be deleted EXCEPT for `metadata.pkl` and `overlaps.pkl`.
4. The resulting dataset should contain these files:
   - `data.hdf5`
   - `data.json`
   - `metadata.pkl`
   - `overlaps.pkl`


# Dataset Format

The on-disk format for the dataset should look like this:

```
DATASET_NAME
|-- metadata.yaml (optional)
|-- metadata.pkl
|-- c_00000
    |-- images
        |-- 0000.{jpg,png}
        |-- ...
        |-- 0099.{jpg,png}
    |-- coords{.npz}
        |-- 0000.npy
        |-- ...
        |-- 0099.npy
    |-- normals{.npz}
        |-- 0000.npy
        |-- ...
        |-- 0099.npy
...
|-- c_00099
    |-- ...
```

Instead of individual .npy files, the coordinates and normals can be saved together in a single
zipped .npz archive, see np.savez_compressed. The reason to save them as raw numpy arrays instead
of in an image format like png is to avoid loss of precision by converting coordinates from 32-bit
floating point values to 8-bit integer values.


## Coordinates

Each image has a corresponding coordinate file containing the surface coordinates for each pixel
in world coordinates. Pixels that are background (not part of the object) should be set to
(0, 0, 0). Coordinates are saved as 32-bit floating point.


## Normals

Each image has a corresponding normals file containing the surface normals for each pixel. The
normal vectors should have unit length (unit vectors). Pixels that are background (not part of the
object) should be set to (0, 0, 0). Normals are saved as 32-bit floating point.


## Metadata

The metadata has the following structure (shown here in yaml format):

```yaml
params:
  param1: value1
  param2: value2
  ...
objects:
  c_00000:
    0000:
      camera_position:  # a list with the (x, y, z) coordinates
      - x
      - y
      - z
      ...
    0001:
      ...
    ...
  ...
```

The metadata can be saved as a yaml file (convenient for incremental saving during dataset generation),
but a pickled dictionary will need to be saved at some point due to the slow loading of large yaml files.

### params

Contains any global parameters used for generating the dataset. These describe the parameters for
reproducing the dataset. Could include things like the number of objects and images per object,
sampling ranges, and random seed.

### objects

Contains parameters specific to each image. The key thing we need is camera_position---the 3D
location of the camera---which is used for shading during retexturing.
