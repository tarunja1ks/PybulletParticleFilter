import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Load the occupancy grid map
ogm_data = np.load('downsampled_map.npy')

class PathDrawer:
    def __init__(self, ax, fig):
        self.ax = ax
        self.fig = fig
        self.path = []
        self.line, = ax.plot([], [], marker="o", color="red")  # Initialize the line
        self.dragging = False
        self.cid_press = fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_release = fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        self.dragging = True

    def on_motion(self, event):
        if not self.dragging or event.inaxes != self.ax:
            return
        self.path.append((event.xdata, event.ydata))
        x, y = zip(*self.path)
        self.line.set_data(x, y)
        self.fig.canvas.draw()

    def on_release(self, event):
        self.dragging = False

    def save_path(self, event):
        # Adjust coordinates to center at (0,0)
        centered_path = [(x - 25, y - 25) for x, y in self.path]
        with open('path_coordinates.txt', 'w') as f:
            for x, y in centered_path:
                f.write(f"{y},{x}\n")

    def clear_path(self, event):
        self.path = []
        self.line.set_data([], [])
        self.fig.canvas.draw()

fig, ax = plt.subplots()
ax.imshow(ogm_data, cmap='gray', vmin=-1, vmax=100)
plt.axis('off')

path_drawer = PathDrawer(ax, fig)

ax_save = plt.axes([0.7, 0.05, 0.1, 0.075])
btn_save = Button(ax_save, 'Save Path')
btn_save.on_clicked(path_drawer.save_path)

ax_clear = plt.axes([0.81, 0.05, 0.1, 0.075])
btn_clear = Button(ax_clear, 'Clear Path')
btn_clear.on_clicked(path_drawer.clear_path)

plt.show()
