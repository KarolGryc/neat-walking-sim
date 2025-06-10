from Box2D import b2PolygonShape, b2CircleShape
from WalkerInfo import WalkerInfo
import math

class Walker:
    MAX_JOINT_SPEED = 2 * math.pi * 0.6
    MAX_JOINT_TORQUE = 8

    def __init__(self, position, simulation):
        self.simulation = simulation

        self.energySpent = 0.0
        self._build(position)

        self.left_leg_forward = True
        self.num_steps = 0

    def _build(self, position):
        x, y = position
        self.startX = x

        HEAD_POS = (x, y + 1)

        HIP_RADIUS = 0.1

        LEFT_HIP_POS = (x-HIP_RADIUS, y+0.05)
        RIGHT_HIP_POS = (x+HIP_RADIUS, y+0.05)

        LEFT_KNEE_POS = (x-HIP_RADIUS, y - 0.5)
        RIGHT_KNEE_POS = (x+HIP_RADIUS, y - 0.5)

        LEFT_FOOT_POS = (x-HIP_RADIUS, y - 1)
        RIGHT_FOOT_POS = (x+HIP_RADIUS, y - 1)

        HIP_FORWARD_LIMIT = math.radians(80)
        HIP_BACKWARD_LIMIT = math.radians(30)
        KNEE_FORWARD_LIMIT = math.radians(0)
        KNEE_BACKWARD_LIMIT = math.radians(100)

        # Create the upper legs
        self.left_upper = self._create_limb(LEFT_KNEE_POS, LEFT_HIP_POS, radius=0.1, width=0.2)
        self.right_upper = self._create_limb(RIGHT_KNEE_POS, RIGHT_HIP_POS, radius=0.1, width=0.2)

        # Create the lower legs
        self.left_lower = self._create_limb(LEFT_FOOT_POS, LEFT_KNEE_POS, radius=0.1, width=0.15)
        self.right_lower = self._create_limb(RIGHT_FOOT_POS, RIGHT_KNEE_POS, radius=0.1, width=0.15)

        # Create the torso
        self.torso = self._create_limb(HEAD_POS, position, radius=0.3, width=0.3)

        # Create the joints
        self.left_hip_joint = self.simulation.world.CreateRevoluteJoint(
            bodyA=self.left_upper,
            bodyB=self.torso,
            anchor=LEFT_HIP_POS,
            enableLimit=True,
            lowerAngle=-HIP_FORWARD_LIMIT,
            upperAngle=HIP_BACKWARD_LIMIT,
        )

        self.right_hip_joint = self.simulation.world.CreateRevoluteJoint(
            bodyA=self.right_upper,
            bodyB=self.torso,
            anchor=RIGHT_HIP_POS,
            enableLimit=True,
            lowerAngle=-HIP_FORWARD_LIMIT,
            upperAngle=HIP_BACKWARD_LIMIT,
        )

        self.left_knee_joint = self.simulation.world.CreateRevoluteJoint(
            bodyA=self.left_upper,
            bodyB=self.left_lower,
            anchor=LEFT_KNEE_POS,
            enableLimit=True,
            lowerAngle=-KNEE_BACKWARD_LIMIT,
            upperAngle=KNEE_FORWARD_LIMIT,
        )

        self.right_knee_joint = self.simulation.world.CreateRevoluteJoint(
            bodyA=self.right_upper,
            bodyB=self.right_lower,
            anchor=RIGHT_KNEE_POS,
            enableLimit=True,
            lowerAngle=-KNEE_BACKWARD_LIMIT,
            upperAngle=KNEE_FORWARD_LIMIT,
        )

        for joint in self._joints():
            joint.motorEnabled = True
            joint.motorSpeed = 0
            joint.maxMotorTorque = self.MAX_JOINT_TORQUE

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
            friction=friction,
            groupIndex=-1
        )

        circle_shape = b2CircleShape(radius=radius)
        body.CreateFixture(
            shape=circle_shape,
            density=density,
            friction=friction,
            groupIndex=-1
        )

        body.ResetMassData()

        return body
    
    def update(self, dt, joint_efforts):
        assert len(joint_efforts) == 4, "Expected 4 joint efforts for the walker."
    
        for (effort, joint) in zip(joint_efforts, self._joints()):
            clamped_effort = min(1, max(-1, effort * 2 - 1))
            self.energySpent += abs(clamped_effort) * dt
            joint.motorSpeed = self.MAX_JOINT_SPEED * clamped_effort
        
        new_left_leg_forward = self.left_lower.position[0] < self.right_lower.position[0]
        if new_left_leg_forward != self.left_leg_forward:
            self.left_leg_forward = new_left_leg_forward
            self.num_steps += 1

    def info(self):
        distance = min([b.position[0] for b in self._bodies()])

        return WalkerInfo(
            name="Stefan",
            headAltitude=self.torso.position[1],
            hDistance=distance-self.startX,
            hSpeed=self.torso.linearVelocity[0],
            torsoAngle=self.torso.angle,
            lHipAngle=self.left_hip_joint.angle,
            rHipAngle=self.right_hip_joint.angle,
            lKneeAngle=self.left_knee_joint.angle,
            rKneeAngle=self.right_knee_joint.angle,
            energySpent=self.energySpent,
            stepsTaken=self.num_steps
        )

    def fitness(self):
        is_tipped_over = self.torso.position[1] < 0.5
        is_left_knee_on_ground = self.left_upper.position[1] < 0.2
        is_right_knee_on_ground = self.right_upper.position[1] < 0.2

        info = self.info()

        multiplier = 1.0
        if is_tipped_over:
            multiplier *= 0.05
        if is_left_knee_on_ground:
            multiplier *= 0.5
        if is_right_knee_on_ground:
            multiplier *= 0.5

        fitness = multiplier * info.hDistance - info.energySpent * 0.3 + multiplier * info.stepsTaken * 0.01
        return fitness
    
    def destroy(self):
        for joint in self._joints():
            self.simulation.world.DestroyJoint(joint)

        for body in [self.left_upper, self.right_upper, self.left_lower, self.right_lower, self.torso]:
            self.simulation.world.DestroyBody(body)
    
    def _joints(self):
        return [
            self.left_hip_joint,
            self.right_hip_joint,
            self.left_knee_joint,
            self.right_knee_joint
        ]
    
    def _bodies(self):
        return [
            self.left_upper,
            self.right_upper,
            self.left_lower,
            self.right_lower,
            self.torso
        ]
