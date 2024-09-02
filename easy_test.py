import pinocchio
import numpy as np
from numpy.linalg import norm, solve
from utils import *


urdf_filename = "./urdf/wow_body_large.urdf"
model = pinocchio.buildModelFromUrdf(urdf_filename)
data = model.createData()

# 输出每个帧的 ID 和名称
for frame_id in range(model.nframes):
    frame = model.frames[frame_id]
    print(f"Frame ID: {frame_id}, Name: {frame.name}")
    
for joint_id in range(model.njoints):
    joint = model.joints[joint_id]
    
    # 获取关节名称
    joint_name = model.names[joint_id]
    
    # 获取关节的姿态
    oMi = data.oMi[joint_id]
    position = oMi.translation
    orientation = oMi.rotation
    
    print(f"Joint ID: {joint_id}, Name: {joint_name}")

JOINT_ID = model.getFrameId('left_wrist_roll_joint')
JOINT_ID_2 = model.getFrameId('left_wrist_roll_link')
print(JOINT_ID)
print(JOINT_ID_2)

cj = pinocchio.neutral(model)
print(cj)

pinocchio.forwardKinematics(model, data, cj)
pinocchio.updateFramePlacements(model, data)


print(data.oMf[JOINT_ID])
print(data.oMf[JOINT_ID_2])




exit()

JOINT_ID = 18

eps = 5e-3
IT_MAX = 10000
DT = 1e-3
damp = 1e-12
INTERPOLATE = 1
xyzrpy = [0.5, 0.0, 0.6, -179.99554936501835, -0.050371084658851845, -58.264042381054885]

qup = model.upperPositionLimit.tolist()
qlow = model.lowerPositionLimit.tolist()
qup = np.array(qup)
qlow = np.array(qlow)
cj = np.random.rand(len(qlow)) * (qup - qlow) + qlow
cj = []
# cj = pinocchio.neutral(model)
# print(cj)
# cj[3] = -1.57
# cj = np.zeros(7)
cj = np.array([2.39279073, -0.31706886, -2.45564179, -0.37443547, -0.82545601,  0.476822, 2.02444053])

pinocchio.forwardKinematics(model, data, cj)
pinocchio.updateFramePlacements(model, data)

coMi = data.oMf[JOINT_ID]
origin_xyz = coMi.translation 
# origin_xyz = data.oMf[JOINT_ID]
# + np.array([0, 0, -0.1])
print("start: ", coMi)
print("cj: ", tuple(cj))
print("t: ", tuple(coMi.translation))
print("R: ", rotation_matrix_to_rpy(coMi.rotation))


print(xyzrpy[:3], xyzrpy[3:])
pos = np.array(xyzrpy[:3])
rot = rpy_to_rotation_matrix(*xyzrpy[3:], use_degree=True)
oMdes = pinocchio.SE3(rot, pos)



interp_seq = []
ratio = 1. / INTERPOLATE
for i in range(1, INTERPOLATE+1):
    alpha = i * 1
    interp_seq.append(pinocchio.SE3.Interpolate(coMi, oMdes, alpha))


for target in interp_seq:
    # target.rotation[target.rotation < 1e-12] = 0
    print("target: ", target)

    while True:
        pinocchio.forwardKinematics(model, data, cj)
        pinocchio.updateFramePlacements(model, data)
        
        iMd = data.oMf[JOINT_ID].actInv(target)
        err = pinocchio.log(iMd).vector  # in joint frame

        if norm(err) < eps:
            print("success")
            success = True
            break
        if i >= IT_MAX:
            print(i)
            print("unsuccessful")
            success = False
            break
        J = pinocchio.computeFrameJacobian(model, data, cj, JOINT_ID)
        J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
        v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
        cj = pinocchio.integrate(model, cj, v * DT)
        cj = np.clip(cj, qlow, qup)
        # if not i % 10:
        #     print("%d: error = %s" % (i, err.T))
        i += 1

    
    print(cj)
    for i in range(len(cj)):
        if cj[i] > qup[i] or cj[i] < qlow[i]:
            print(i, cj[i], qup[i], qlow[i])
            raise ValueError

pinocchio.forwardKinematics(model, data, cj)
pinocchio.updateFramePlacements(model, data)
print("after: ", JOINT_ID, data.oMf[JOINT_ID])
print(len(data.oMf))
# print("after: ", JOINT_ID-1, data.oMf[JOINT_ID-1])
# print(len(data.oMi), len(data.iMf))
# print(data.oMi[0], data.iMf[0])

# print("oMi: ", JOINT_ID, data.oMi[7])
# print("iMf: ", JOINT_ID, data.iMf[7])

# for i in range(len(data.oMi)):
#     print()
#     print(i)
#     print(data.oMi[i], data.iMf[i])
#     if i > 0:
#         print(cj[i-1])