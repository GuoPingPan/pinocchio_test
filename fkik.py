import pinocchio
from sys import argv
from os.path import dirname, join, abspath
import numpy as np

# This path refers to Pinocchio source code but you can define your own directory here.
pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "models")

# You should change here to set up your own URDF file or just pass it as an argument of this example.
urdf_filename = "/home/a4090/ikpy/resources/wow/up_body.urdf"

# Load the urdf model
model = pinocchio.buildModelFromUrdf(urdf_filename)
print("model name: " + model.name)

# Create data required by the algorithms
data = model.createData()

# Sample a random configuration
q = pinocchio.randomConfiguration(model)
print("q: %s" % q.T)
print("q: ", np.array2string(q, separator=', '))
q = np.array([ 1.65344199, -0.82954902,  1.41549612,  1.04454012,  2.46988415,
 -1.05857021, -0.57672035,  1.60937757, -1.25001049, -0.55825108,
 -0.11301474,  0.45104824, -0.81129316,  0.04690319,  1.58280404,
  2.49717041])

q = np.array([0] * 16)


# Perform the forward kinematics over the kinematic tree
pinocchio.forwardKinematics(model, data, q)

# Print out the placement of each joint of the kinematic tree
# for name, oMi in zip(model.names, data.oMi):
#     print(("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat)))

# print(len(data.oMi))
# print(data.oMi[0])
# print(data.oMi[7])
# print(data.oMi[16])

eps = 1e-3
IT_MAX = 1000
DT = 1e-1
damp = 1e-12
JOINT_ID = 8
from numpy.linalg import norm, solve

i = 0

oMdes = pinocchio.SE3(np.eye(3), np.array([0 + 0.1,  0.187896, -0.293583 + 0.1]))

# print(type(oMdes))

print(data.oMi[JOINT_ID])
print("=" * 10)
for i in range(1, 11):
    alpha = i * 0.1
    print(pinocchio.SE3.Interpolate(data.oMi[JOINT_ID], oMdes, alpha))
print("=" * 10)
print(oMdes)

exit()
while True:
    pinocchio.forwardKinematics(model, data, q)
    iMd = data.oMi[JOINT_ID].actInv(oMdes)
    err = pinocchio.log(iMd).vector  # in joint frame

    if norm(err) < eps:
        success = True
        break
    if i >= IT_MAX:
        success = False
        break
    J = pinocchio.computeJointJacobian(model, data, q, JOINT_ID)  # in joint frame
    J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
    v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
    q = pinocchio.integrate(model, q, v * DT)
    if not i % 10:
        print("%d: error = %s" % (i, err.T))
    i += 1
    
if success:
    print("Convergence achieved!")
else:
    print(
        "\nWarning: the iterative algorithm has not reached convergence to the desired precision"
    )

print("\nresult: %s" % q.flatten().tolist())
print("\nfinal error: %s" % err.T)
