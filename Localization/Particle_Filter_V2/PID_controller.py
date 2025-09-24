from math import pi

class PID:
    def __init__(self, Kp=1.2, Ki=1, Kd=0.001):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.I_term = 0.0
        
        self.last_error = 0.0
        
    def adjust(self, feedback, reference, delta_time):
        # Determine current error
        error = feedback - reference
        if reference == pi and feedback < 0:
            error = feedback + pi

        if abs(error) <= (pi - abs(error)):
            curr_error = error
        else:
            curr_error = -(pi - abs(error))

        # Determine P, I, D terms to obtain the adjustment for next time step
        P_term = self.Kp * curr_error
        self.I_term = curr_error * delta_time
        D_term = 0.0
        if delta_time > 0:
            D_term = self.Kd * (curr_error-self.last_error) / delta_time

        self.last_error = curr_error

        return P_term + self.Ki * self.I_term + D_term
