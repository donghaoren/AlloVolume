import StringIO
import struct
import numpy as np

def VolumeFrom3DArray(array, pos_min, pos_max, fout):
    """Create volume file from numpy array.
    Args:
        array: 3D numpy array.
        pos_min: 3-tuple of floats containing the minimum world coordinates.
        pos_max: 3-tuple of floats containing the minimum world coordinates.
        fout: Output file (file-like object that can `write`).
    """
    fBlocks = StringIO.StringIO()
    fData = StringIO.StringIO()
    array = array.astype("float32", "F")
    bytes = array.data
    fData.write(bytes)

    fBlocks.write(struct.pack("ffffffiiiiQ",
        pos_min[0], pos_min[1], pos_min[2],
        pos_max[0], pos_max[1], pos_max[2],
        array.shape[0], array.shape[1], array.shape[2],
        1,
        0
    ))
    blocks = fBlocks.getvalue()
    data = fData.getvalue()
    fout.write(struct.pack("Q", 1))
    fout.write(struct.pack("Q", len(data) / 4))
    fout.write(data)
    fout.write(blocks)
    fout.close()
