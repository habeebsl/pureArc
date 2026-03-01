import math

class ReleaseDetector:
    def __init__(self,
                 distance_threshold=15,
                 elbow_velocity_threshold=8,
                 ball_velocity_threshold=-5):
        """
        distance_threshold: min increase in ball-wrist distance per frame
        elbow_velocity_threshold: min elbow extension speed (deg/frame)
        ball_velocity_threshold: negative = moving upward in image coords
        """

        self.prev_distance = None
        self.prev_elbow = None
        self.prev_ball_y = None

        self.distance_threshold = distance_threshold
        self.elbow_velocity_threshold = elbow_velocity_threshold
        self.ball_velocity_threshold = ball_velocity_threshold

    def compute_distance(self, wrist_x, wrist_y, ball_x, ball_y):
        return math.sqrt((ball_x - wrist_x)**2 + (ball_y - wrist_y)**2)

    def detect(self, wrist_x, wrist_y, ball_x, ball_y, elbow_angle):
        release = False

        # Compute current distance
        distance = self.compute_distance(wrist_x, wrist_y, ball_x, ball_y)

        if self.prev_distance is not None and \
           self.prev_elbow is not None and \
           self.prev_ball_y is not None:

            # Compute velocities
            distance_velocity = distance - self.prev_distance
            elbow_velocity = elbow_angle - self.prev_elbow
            ball_velocity = ball_y - self.prev_ball_y  # image coords: up = negative

            # Release logic
            if (distance_velocity > self.distance_threshold and
                elbow_velocity > self.elbow_velocity_threshold and
                ball_velocity < self.ball_velocity_threshold):

                release = True

        # Update previous values
        self.prev_distance = distance
        self.prev_elbow = elbow_angle
        self.prev_ball_y = ball_y

        return release