import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import time

class MultiLinePlot:
    def __init__(self, n_lines=2, window_size=200, update_interval=20):
        """
        n_lines: number of lines on the same graph
        window_size: number of points in rolling window
        update_interval: in milliseconds
        """
        self.n_lines = n_lines
        self.window_size = window_size
        self.update_interval = update_interval

        self.fig, self.ax = plt.subplots(figsize=(6,4))

        # Data buffers
        self.data = [deque([0]*window_size, maxlen=window_size) for _ in range(n_lines)]
        self.time_data = deque([0]*window_size, maxlen=window_size)

        # Line objects
        self.lines = [self.ax.plot(self.time_data, d)[0] for d in self.data]

        self.ax.set_ylim(-1, 1)  # adjust as needed
        self.start_time = time.time()

        self.ani = FuncAnimation(self.fig, self._update, interval=self.update_interval, blit=True)

    def add_data(self, values):
        """
        Add new values for all lines at current time
        values: list of floats, length must equal n_lines
        """
        t = time.time() - self.start_time
        self.time_data.append(t)
        for i, val in enumerate(values):
            self.data[i].append(val)

    def _update(self, frame):
        for i, line in enumerate(self.lines):
            line.set_data(self.time_data, self.data[i])
        self.ax.set_xlim(self.time_data[0], self.time_data[-1])
        return self.lines

    def show(self):
        plt.show()


