from Box2D import b2PolygonShape, b2CircleShape, b2RevoluteJoint
import pygame
import math

class Walker:
    MAX_JOINT_SPEED = 2 * math.pi * 2
    MAX_JOINT_TORQUE = 2

    def __init__(self, position, simulation):
        self.simulation = simulation

        self.circles = []
        self.polygons = []

        self._build(position)

    def _build(self, position):
        x, y = position

        HEAD_POS = (x, y + 1)

        LEFT_HIP_POS = (x - 0.1, y+0.1)
        RIGHT_HIP_POS = (x + 0.1, y+0.1)

        LEFT_KNEE_POS = (x - 0.3, y - 0.5)
        RIGHT_KNEE_POS = (x + 0.3, y - 0.5)

        LEFT_FOOT_POS = (x - 0.3, y - 1)
        RIGHT_FOOT_POS = (x + 0.3, y - 1)

        # Create the upper legs
        left_upper = self._create_limb(LEFT_KNEE_POS, LEFT_HIP_POS, radius=0.1, width=0.15)
        right_upper = self._create_limb(RIGHT_KNEE_POS, RIGHT_HIP_POS, radius=0.1, width=0.15)

        # Create the lower legs
        left_lower = self._create_limb(LEFT_FOOT_POS, LEFT_KNEE_POS, radius=0.1, width=0.15)
        right_lower = self._create_limb(RIGHT_FOOT_POS, RIGHT_KNEE_POS, radius=0.1, width=0.15)

        # Create the torso
        torso = self._create_limb(HEAD_POS, position, radius=0.3, width=0.4)

        # Create the joints
        self.left_hip_joint = self.simulation.world.CreateRevoluteJoint(
            bodyA=left_upper,
            bodyB=torso,
            anchor=LEFT_HIP_POS,
            collideConnected=False,
            enableMotor=True,
            motorSpeed=self.MAX_JOINT_SPEED,
            maxMotorTorque=0,
        )

        self.right_hip_joint = self.simulation.world.CreateRevoluteJoint(
            bodyA=right_upper,
            bodyB=torso,
            anchor=RIGHT_HIP_POS,
            collideConnected=False,
            enableMotor=True,
            motorSpeed=self.MAX_JOINT_SPEED,
            maxMotorTorque=0,
        )

        self.left_knee_joint = self.simulation.world.CreateRevoluteJoint(
            bodyA=left_upper,
            bodyB=left_lower,
            anchor=LEFT_KNEE_POS,
            collideConnected=False,
            enableMotor=True,
            motorSpeed=self.MAX_JOINT_SPEED,
            maxMotorTorque=0,
        )

        self.right_knee_joint = self.simulation.world.CreateRevoluteJoint(
            bodyA=right_upper,
            bodyB=right_lower,
            anchor=RIGHT_KNEE_POS,
            collideConnected=False,
            enableMotor=True,
            motorSpeed=self.MAX_JOINT_SPEED,
            maxMotorTorque=0,
        )


    def _create_limb(self, posA, posB, radius=0.1, width=0.1, density=1.0, friction=0.5):

        dx, dy = posB[0] - posA[0], posB[1] - posA[1]
        length = (dx**2 + dy**2)**0.5
        angle = math.atan2(dy, dx) - math.pi/2

        body = self.simulation.world.CreateDynamicBody(
            position=posA,
        )

        rect_shape = b2PolygonShape()
        rect_shape.SetAsBox(width / 2, length / 2, (dx/2, dy/2), angle)
        body.CreateFixture(
            shape=rect_shape,
            density=density,
            friction=friction
        )

        circle_shape = b2CircleShape(radius=radius)
        body.CreateFixture(
            shape=circle_shape,
            density=density,
            friction=friction
        )

        body.ResetMassData()

        return body
    
    def update(self, joint_efforts):
        if len(joint_efforts) != 4:
            raise ValueError("Expected 4 joint efforts for the walker.")
    
        for (effort, joint) in zip(joint_efforts, [
            self.left_hip_joint,
            self.right_hip_joint,
            self.left_knee_joint,
            self.right_knee_joint
        ]):
            clamped_effort = min(1, max(-1, effort))
            joint.motorSpeed = self.MAX_JOINT_SPEED * (1 if clamped_effort > 0 else -1)
            joint.maxMotorTorque = abs(clamped_effort) * self.MAX_JOINT_TORQUE
