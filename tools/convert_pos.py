import numpy as np

def quat_to_rot(qx, qy, qz, qw):
    # 归一化防止数值误差
    norm = np.linalg.norm([qx, qy, qz, qw])
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),     2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx + zz),     2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx + yy)],
    ], dtype=np.float32)

poses = []
path = "data0/goat-core/"+input("Enter scene name (e.g., 4ok): ")
with open(path+"/local_pos.txt") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 8:
            continue
        _, qx, qy, qz, qw, x, y, z = map(float, parts)
        R_c2w = quat_to_rot(qx, qy, qz, qw)
        R_w2c = np.linalg.inv(R_c2w)
        flip = np.diag([1.0, -1.0, -1.0]).astype(np.float32)  # RUB -> RDF
        R = flip @ R_w2c
        position = np.array([x, y, z], dtype=np.float32)
        t = flip @ position
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R.T
        T[:3, 3] = t
        poses.append(T)

poses = np.stack(poses)  # (N, 4, 4)
print("poses shape:", poses.shape)
print("first pose:\n", poses[0])
np.savetxt(path+"/traj.txt", poses.reshape(len(poses), 16), fmt="%.20e")