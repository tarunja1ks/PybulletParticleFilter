import time

class Timings:
    """
    Timing module for car and sensors controls.
    """
    def __init__(self, fps):
        self.fps = fps
        self.subject_time = time.time()

    def update_time(self):
        curr_subject_time = time.time()

        if (curr_subject_time - self.subject_time > 1.0/self.fps):
            self.subject_time = curr_subject_time
            return True

        return False

