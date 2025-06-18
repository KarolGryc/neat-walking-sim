from Box2D import b2PolygonShape, b2CircleShape, b2RevoluteJoint
from WalkerInfo import WalkerInfo
import math

BRAKE_ON_NO_INPUT = False
max_height_score = 0.0

class Walker:
    MAX_JOINT_SPEED = 2 * math.pi * 0.9
    MAX_JOINT_TORQUE = 12
    global max_height_score

    def __init__(self, position, simulation):
        self.simulation = simulation
        self.dead = False

        self.energySpent = 0.0
        self._build(position)

        self.left_leg_forward = 1
        self.right_leg_forward = 1

    def is_dead(self):
        if self.dead:
            return True

        info = self.info()
        head_height = (info.headAltitude - 0.3)
        if head_height < 0.025:
            self.dead = True
        
        return self.dead

    def _build(self, position):
        x, y = position
        self.startX = x

        self._height_score = 0.0
        self._total_time = 0.0

        HEAD_POS = (x, y + 1)

        HIP_RADIUS = 0.0

        LEFT_HIP_POS = (x-HIP_RADIUS, y+0.05)
        RIGHT_HIP_POS = (x+HIP_RADIUS, y+0.05)

        LEFT_KNEE_POS = (x-HIP_RADIUS, y - 0.5)
        RIGHT_KNEE_POS = (x+HIP_RADIUS, y - 0.5)

        LEFT_FOOT_POS = (x-HIP_RADIUS, y - 1)
        RIGHT_FOOT_POS = (x+HIP_RADIUS, y - 1)

        HIP_FORWARD_LIMIT = math.radians(120)
        HIP_BACKWARD_LIMIT = math.radians(-25)
        KNEE_FORWARD_LIMIT = math.radians(-6)
        KNEE_BACKWARD_LIMIT = math.radians(160)

        LEFT_COLOR = (0, 100, 255)

        # Create the upper legs
        self.left_upper = self._create_limb(LEFT_KNEE_POS, LEFT_HIP_POS, radius=0.1, width=0.2, color=LEFT_COLOR)
        self.right_upper = self._create_limb(RIGHT_KNEE_POS, RIGHT_HIP_POS, radius=0.1, width=0.2)

        # Create the lower legs
        self.left_lower = self._create_limb(LEFT_FOOT_POS, LEFT_KNEE_POS, radius=0.125, width=0.15, color=LEFT_COLOR)
        self.right_lower = self._create_limb(RIGHT_FOOT_POS, RIGHT_KNEE_POS, radius=0.125, width=0.15)

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
            if BRAKE_ON_NO_INPUT:
                joint.maxMotorTorque = self.MAX_JOINT_TORQUE
            else:
                joint.maxMotorTorque = 0

    def _create_limb(self, posA, posB, radius=0.1, width=0.1, friction=0.5, color=(0, 150, 255)):
        dx, dy = posB[0] - posA[0], posB[1] - posA[1]
        length = math.sqrt(dx * dx + dy * dy)
        angle = math.atan2(dy, dx) - math.pi/2

        body = self.simulation.world.CreateDynamicBody(
            position=posA,
            userData={'color': color}
        )

        rect_shape = b2PolygonShape()
        rect_shape.SetAsBox(width / 2, length / 2, (dx/2, dy/2), angle)
        body.CreateFixture(
            shape=rect_shape,
            density=1.0,
            friction=friction,
            groupIndex=-1
        )

        circle_shape = b2CircleShape(radius=radius)
        body.CreateFixture(
            shape=circle_shape,
            density=1.0,
            friction=friction,
            groupIndex=-1
        )

        body.ResetMassData()

        return body
    
    def update(self, dt, joint_efforts):
        # self._total_time += dt
        # self._height_score += self.torso.position[1] * dt

        for (effort, joint) in zip(joint_efforts, self._joints()):
            clamped_effort = min(1, max(-1, effort * 2 - 1))
            joint.motorSpeed = self.MAX_JOINT_SPEED * (1 if clamped_effort > 0 else -1)
            joint.maxMotorTorque = abs(float(clamped_effort)) * self.MAX_JOINT_TORQUE

    def info(self):
        # distance = min(b.position[0] for b in self._bodies())
        distance = self.torso.position[0]
        return WalkerInfo(
            headAltitude=self.torso.position[1],
            hDistance=distance-self.startX,
            hSpeed=self.torso.linearVelocity[0],
            torsoAngle=self.torso.angle,
            lHipAngle=self.left_hip_joint.angle,
            rHipAngle=self.right_hip_joint.angle,
            lKneeAngle=self.left_knee_joint.angle,
            rKneeAngle=self.right_knee_joint.angle,
            energySpent=self.energySpent,
            lHipSpeed=self.left_hip_joint.speed,
            rHipSpeed=self.right_hip_joint.speed,
            lKneeSpeed=self.left_knee_joint.speed,
            rKneeSpeed=self.right_knee_joint.speed,
        )

    def fitness(self):
        info = self.info()

        # normalized_height_score = (self._height_score) /  self._total_time
        # average_speed_score = info.hDistance / self._total_time
        # fitness = (average_speed_score * 0.3 if average_speed_score > 0 else 0) ** (normalized_height_score if normalized_height_score > 0 else 0.000001) 
        # HEIGHT_WEIGHT = 1
        # DISTANCE_WEIGHT = 1
        # fitness = HEIGHT_WEIGHT * self._height_score / self._total_time + DISTANCE_WEIGHT * info.hDistance / self._total_time
        fitness = info.hDistance
        # print(f"{normalized_height_score:4f}", )
        return fitness

    # If possible just create new world for walkers    
    def destroy(self):
        for joint in self._joints():
            self.simulation.world.DestroyJoint(joint)

        for body in [self.left_upper, self.right_upper, self.left_lower, self.right_lower, self.torso]:
            self.simulation.world.DestroyBody(body)
    
    def _joints(self):
        return (
            self.left_hip_joint,
            self.right_hip_joint,
            self.left_knee_joint,
            self.right_knee_joint
        )
    
    def _bodies(self):
        return (
            self.left_upper,
            self.right_upper,
            self.left_lower,
            self.right_lower,
            self.torso
        )
