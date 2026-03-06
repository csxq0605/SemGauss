import numpy as np

def quaternion_to_rotation_matrix(q):
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    R = np.array(
            [
                [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
                [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2],
            ]
        )
    return R

poses = []
path = "data0/goat-core/"+input("Enter scene name (e.g., office0): ")
pose_path = path+"/local_pos.txt"
num_imgs = len(open(pose_path).readlines())
with open(pose_path, "r") as f:
    lines = f.readlines()
for i in range(num_imgs):
            line = lines[i]
            line = line.split()
            index = line[0]
            q = line[1:5]
            position = line[5:]
            q = np.array([float(x) for x in q])

            R_c2w = quaternion_to_rotation_matrix(q)
            R_w2c = np.linalg.inv(R_c2w)

            R_RUB_to_RDF = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            R = R_RUB_to_RDF @ R_w2c

            position = np.array([float(x) for x in position])
            t = R_RUB_to_RDF @ position

            c2w = np.eye(4)
            c2w[:3, :3] = R.T
            c2w[:3, 3] = t

            poses.append(c2w)

poses = np.stack(poses)  # (N, 4, 4)
print("poses shape:", poses.shape)
print("first pose:\n", poses[0])
np.savetxt(path+"/traj.txt", poses.reshape(len(poses), 16), fmt="%.20e")