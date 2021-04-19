import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time

class ArmHitEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 100
    }
    initial_positions = {
        "joint_1" : 0,
        "joint_2" : 0,
        "joint_3" : 0,
        "joint_4" : 0,
        "joint_5" : 0,
        "joint_6" : 0
    }
    def __init__(self):
        # self.action_space = spaces.Box(low=np.array([-np.pi,-2.01,-0.69,-np.pi,-0.78,-np.pi]),high=np.array([np.pi,2.01,3.83,np.pi,3.92,np.pi]))
        # self.observation_space = 
        self._timeStep = 0.5
        self._maxVelocity = np.array([0.8, 0.7, 0.8, 1.0, 1.0, 1.0])
        self.state = None
        self._joint_name_to_ids = {}
        p.connect(p.GUI)
        # p.setRealTimeSimulation(1)
        self.reload_robot()
        self.end_eff_idx = 7
        self._target = 1
        self.targetPos = [[0.26,0.15,0.08], [0.28,-0.24,0.08]]
    
    def reload_robot(self):
        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION
        p.resetSimulation()
        p.setRealTimeSimulation(1)
        p.setGravity(0,0,-9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # self.collison_box_id = p.createCollisionShape(shapeType=p.GEOM_BOX,halfExtents=[60, 5, 5])

        planeUid=p.loadURDF("plane.urdf",basePosition=[0,0,0.022])
        rest_poses=[0,0,0,0,0,0]
        self.robot_id=p.loadURDF("arm_env/probot_anno.urdf",useFixedBase=True, flags=flags)
        robotStartOrientation = p.getQuaternionFromEuler([0,0,0])
        robotEndOrientation = p.getQuaternionFromEuler([1.57,0,1.57])
        robotStartPos = [0,0,0]
        p.resetBasePositionAndOrientation(self.robot_id,robotStartPos,robotStartOrientation)  

        self.ll, self.ul, self.jr, self.rs = self.get_joint_ranges()
        self.create_collision()
        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode("UTF-8")
            joint_type = joint_info[2]

            if joint_type is p.JOINT_REVOLUTE or joint_type is p.JOINT_PRISMATIC:
                assert joint_name in self.initial_positions.keys()

                self._joint_name_to_ids[joint_name] = i

                p.resetJointState(self.robot_id, i, self.initial_positions[joint_name])
                p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL,
                                        targetPosition=self.initial_positions[joint_name])
        time.sleep(1)
        self._target = 1
        # self.hit = 0
        # print(self.jointIndex)
        self.read_state()
    
    def create_collision(self):
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=0.03,
            length=0.22,
            rgbaColor=[1,0,0,1]
        )

        collison_box_id = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=0.03,
            height=0.22
        )

        self.wall_id = p.createMultiBody(
            baseMass=10000,
            baseCollisionShapeIndex=collison_box_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[0.27, 0, 0.11]
        )

        ring_id1 = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.02,
            rgbaColor=[0,0,1,1]
        )

        collison_ring_id1 = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.02
        )

        self.ring_1 = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collison_ring_id1,
            baseVisualShapeIndex=ring_id1,
            basePosition=[0.26,0.15,0.08]
        )

        self.ring_2 = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collison_ring_id1,
            baseVisualShapeIndex=ring_id1,
            basePosition=[0.28,-0.24,0.08]
        )
    
    def get_joint_ranges(self):
        lower_limits, upper_limits, joint_ranges, rest_poses = [], [], [], []

        for joint_name in self._joint_name_to_ids.keys():
            jointInfo = p.getJointInfo(self.robot_id, self._joint_name_to_ids[joint_name])

            ll, ul = jointInfo[8:10]
            jr = ul - ll
            # For simplicity, assume resting state == initial state
            rp = self.initial_positions[joint_name]
            lower_limits.append(ll)
            upper_limits.append(ul)
            joint_ranges.append(jr)
            rest_poses.append(rp)

        return lower_limits, upper_limits, joint_ranges, rest_poses
    
    def check_collision(self, obj_id):
        # check if there is any collision with an object

        contact_pts = p.getContactPoints(obj_id, self.robot_id)
        # check if the contact is on the fingertip(s)

        return len(contact_pts) > 0

    def read_state(self):
        jointStates = p.getJointStates(self.robot_id, self._joint_name_to_ids.values())
        jointPoses = [x[0] for x in jointStates]
        jointVelocity = [x[1] for x in jointStates]
        self.state = np.hstack((np.array(jointPoses), np.array(jointVelocity)))
        self.state = np.append(self.state, self._target)
        # P_min, P_max = p.getAABB(robot_id)


    def reset(self):
        # p.resetSimulation()
        self.reload_robot()
        return self.state
        # self.state = p.getLinkState(self.pandaUid, 6)
    
    def outside(self):
        linkstate = p.getLinkState(self.robot_id, self.end_eff_idx)
        if linkstate[0][2] > 0.5:
            return True
        if linkstate[0][0] < 0:
            return True
        return False
    
    def step(self, action):
        # p.setJointMotorControlArray(self.pandaUid, self.jointIndex, p.VELOCITY_CONTROL, targetVelocities = action)
        # for i in range(len(self.jointIndex)):
        # action = p.calculateInverseKinematics(self.robot_id, self.end_eff_idx, self.targetPos[self.target-1])
        action = action*self._maxVelocity
        p.setJointMotorControlArray(self.robot_id, self._joint_name_to_ids.values(), p.VELOCITY_CONTROL, targetVelocities=action)
        # p.setJointMotorControlArray(self.robot_id, self._joint_name_to_ids.values(), p.POSITION_CONTROL, targetPositions=action)
        time.sleep(0.01)
        self.read_state()
        # for i in 
        done = False
        reward = -1.0
        if self.check_collision(self.wall_id) or self.outside():
            reward -= 100
            done = True
        elif self.check_collision(self.ring_1):
            if self._target == 1:
                reward += 100
                self._target = 2
            else:
                reward -= 10
        elif self.check_collision(self.ring_2):
            if self._target == 2:
                reward += 100
                self._target = 1
            else:
                reward -= 10

        return self.state, reward, done, {}
    
        

