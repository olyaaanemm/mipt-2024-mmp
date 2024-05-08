import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
from math import pi
from matplotlib.animation import FuncAnimation
from client_view import * 

# Scaling factor for distance, m-1. The box dimension is therefore 1/rscale.
rscale = 5.e6
# Scale time by this factor, in s-1.
tscale = 1e9    # i.e. time will be measured in nanoseconds.

# Time step in scaled time units.
FPS = 1_000
dt = 1/FPS

sbar = 0.0

r = 1.40126e-7 * rscale
m = 6.645e-27

fe_charge = 26 # electrons
u_charge = 90 # electrons

k_e = 3 

# === Particle features ===
# Number of particles.
n = 1_024
theta = np.random.random(n) * 2 * np.pi
# =========================

# Initialize the particles' positions randomly.
# pos = np.random.random((n, 2))
x = np.linspace(0.0,1.0,32)
y = np.linspace(0.0,1.0,32)

pos = []

for i in range(32): 
    for j in range(32): 
        pos.append([x[i], y[j]])

pos = np.array(pos)

# pos = np.random.uniform(low=0, high=1, size=(n , 2)) 

point = np.array([[0.5, 1.0]])

# Initialize the particles velocities with random orientations and random
# magnitudes  around the mean speed, sbar.
s0 = sbar * np.random.random(n)
vel = (s0 * np.array((np.cos(theta), np.sin(theta)))).T
point_vel = [[0.0, -1 * 300 * rscale / tscale]]

sim = MDSimulation(pos_1=pos, 
                   pos_2=point, 
                   vel_1=vel, 
                   vel_2= point_vel, 
                   r_1=r, 
                   r_2=r * 10, 
                   m_ch= -1.6e-19, 
                   env_ch= -fe_charge * 1.6e-19)

# Set up the Figure and make some adjustments to improve its appearance.
DPI = 100
width, height = 500, 500
fig = plt.figure(figsize=(width/DPI, height/DPI), dpi=DPI)
fig.subplots_adjust(left=0, right=0.97)
sim_ax = fig.add_subplot(aspect='equal', autoscale_on=False)
sim_ax.set_xticks([])
sim_ax.set_yticks([])
# Make the box walls a bit more substantial.
for spine in sim_ax.spines.values():
    spine.set_linewidth(2)

#speed_ax = fig.add_subplot(122)
#speed_ax.set_xlabel('Speed $v\,/m\,s^{-1}$')
#speed_ax.set_ylabel('$f(v)$')

particles_1, = sim_ax.plot([], [], 'ro')
particles_2, = sim_ax.plot([], [], 'go--')

#speeds = get_speeds(sim.vel_1)

#speed_hist = Histogram(speeds, 2 * sbar_1, 80, density=True)
#speed_hist.draw(speed_ax)
#speed_ax.set_xlim(speed_hist.left[0], speed_hist.right[-1])
#ticks = np.linspace(0, 600, 7, dtype=int)
#speed_ax.set_xticks(ticks * rscale/tscale)
#speed_ax.set_xticklabels([str(tick) for tick in ticks])
#speed_ax.set_yticks([])

fig.tight_layout()

# The 2D Maxwell-Boltzmann equilibrium distribution of speeds.
# mean_KE = get_KE(speeds, m) / n
# a = sim.m / 2 / mean_KE
# Use a high-resolution grid of speed points so that the exact distribution looks smooth.
#sgrid_hi = np.linspace(0, speed_hist.bins[-1], 200)

# ==== expected density line ====
#f = 2 * a * sgrid_hi * np.exp(-a * sgrid_hi**2)
#mb_line, = speed_ax.plot(sgrid_hi, f, c='0.7')
# Maximum value of the 2D Maxwell-Boltzmann speed distribution.
#fmax = np.sqrt(sim.m_1 / mean_KE / np.e)
#speed_ax.set_ylim(0, fmax)

# For the distribution derived by averaging, take the abcissa speed points from the centre of the histogram bars.
#sgrid = (speed_hist.bins[1:] + speed_hist.bins[:-1]) / 2
#mb_est_line, = speed_ax.plot([], [], c='r')
#mb_est = np.zeros(len(sgrid))

# A text label indicating the time and step number for each animation frame.
#xlabel, ylabel = sgrid[-1] / 2, 0.8 * fmax
#label = speed_ax.text(xlabel, ylabel, '$t$ = {:.1f}s, step = {:d}'.format(0, 0),
                      #backgroundcolor='w')

def init_anim():
    """Initialize the animation"""
    particles_1.set_data([], [])
    particles_2.set_data([], [])

    return particles_1, particles_2 #speed_hist.patch, mb_est_line, label

def animate(i):
    """Advance the animation by one step and update the frame."""
    global sim, verts, mb_est_line, mb_est
    sim.advance(dt)
    
    particles_1.set_data(sim.pos_1[:, X], sim.pos_1[:, Y])
    particles_1.set_color('g')
    particles_2.set_data(sim.pos_2[:, X], sim.pos_2[:, Y])
    particles_2.set_color('r')
    particles_1.set_markersize(0.5)
    particles_2.set_markersize(2)

    #speeds = get_speeds(sim.vel_1)
    #speed_hist.update(speeds)

    # Once the simulation has approached equilibrium a bit, start averaging
    # the speed distribution to indicate the approximation to the Maxwell-
    # Boltzmann distribution.
    #if i >= IAV_START:
    #    mb_est += (speed_hist.hist - mb_est) / (i - IAV_START + 1)
    #    mb_est_line.set_data(sgrid, mb_est)

    #label.set_text('$t$ = {:.1f} ns, step = {:d}'.format(i*dt, i))

    return particles_1, particles_2 #speed_hist.patch, mb_est_line, label

# Only start averaging the speed distribution after frame number IAV_ST.
IAV_START = 20
# Number of frames; set to None to run until explicitly quit.
frames = 1000
anim = FuncAnimation(fig, animate, frames=frames, interval=10, blit=False,
                    init_func=init_anim)

plt.show()

