from scipy.spatial.distance import pdist, squareform
import matplotlib.path as path
import matplotlib.patches as patches
import numpy as np


# axes
X, Y = 0, 1
k_e = 3 * 10**9

class MDSimulation:

    def __init__(self, pos_1, pos_2, vel_1, vel_2, r_1, r_2, env_ch, m_ch):
        """
        Initialize the simulation with identical, circular particles of radius
        r and mass m. The n x 2 state arrays pos and vel hold the n particles'

        """

        self.pos_1 = np.asarray(pos_1, dtype=float)
        self.pos_2 = np.asarray(pos_2, dtype=float)
        self.vel_1 = np.asarray(vel_1, dtype=float)
        self.vel_2 = np.asarray(vel_2, dtype=float)
        self.n = self.pos_1.shape[0] + self.pos_2.shape[0]
        self.m = self.pos_1.shape[0]
        self.r_1 = r_1
        self.r_2 = r_2
        self.env_ch = env_ch
        self.m_ch = m_ch
        self.nsteps = 0


    def advance(self, dt):
        """Advance the simulation by dt seconds."""

        self.nsteps += 1
        # Update the particles' positions according to their velocities.
        self.pos_2 += self.vel_2 * dt
        # Find indices for all unique collisions.
        dist = squareform(pdist(np.concatenate((self.pos_1, self.pos_2))))
        iarr, jarr = np.where(dist < 2 * self.r_1)
        k = iarr < jarr
        iarr, jarr = iarr[k], jarr[k]

        # For each collision, update the velocities of the particles involved.
        for i, j in zip(iarr, jarr):
            
            pos_i, vel_i =  (self.pos_1[i], self.vel_1[i]) if i < self.m else (self.pos_2[i - self.m], self.vel_2[i - self.m])
            pos_j, vel_j =  (self.pos_1[j], self.vel_1[j]) if j < self.m else (self.pos_2[j - self.m], self.vel_2[j - self.m])
            
            rel_pos, rel_vel = pos_i - pos_j, vel_i - vel_j
            r_rel = rel_pos @ rel_pos
            v_rel = rel_vel @ rel_pos
            
            if i >= self.m:
                self.vel_2[i - self.m] = self.vel_2[i - self.m] + (k_e * self.env_ch * self.m_ch * rel_pos * dt) / (np.sqrt(r_rel) ** 3 * 1.4e-28)

            if j >= self.m:
                self.vel_2[j - self.m] = self.vel_2[j - self.m] + (k_e * self.env_ch * self.m_ch * rel_pos * dt) / (np.sqrt(r_rel) ** 3 * 1.4e-28)
                


class Histogram:
    """A class to draw a Matplotlib histogram as a collection of Patches."""

    def __init__(self, data, xmax, nbars, density=False):
        """Initialize the histogram from the data and requested bins."""
        self.nbars = nbars
        self.density = density
        self.bins = np.linspace(0, xmax, nbars)
        self.hist, bins = np.histogram(data, self.bins, density=density)
        
        self.left = np.array(bins[:-1])
        self.right = np.array(bins[1:])
        self.bottom = np.zeros(len(self.left))
        self.top = self.bottom + self.hist
        nrects = len(self.left)
        self.nverts = nrects * 5
        self.verts = np.zeros((self.nverts, 2))
        self.verts[0::5, 0] = self.left
        self.verts[0::5, 1] = self.bottom
        self.verts[1::5, 0] = self.left
        self.verts[1::5, 1] = self.top
        self.verts[2::5, 0] = self.right
        self.verts[2::5, 1] = self.top
        self.verts[3::5, 0] = self.right
        self.verts[3::5, 1] = self.bottom

    def draw(self, ax):
        """Draw the histogram by adding appropriate patches to Axes ax."""
        codes = np.ones(self.nverts, int) * path.Path.LINETO
        codes[0::5] = path.Path.MOVETO
        codes[4::5] = path.Path.CLOSEPOLY
        barpath = path.Path(self.verts, codes)
        self.patch = patches.PathPatch(barpath, fc='tab:green', ec='k',
                                  lw=0.5, alpha=0.5)
        ax.add_patch(self.patch)

    def update(self, data):
        """Update the rectangle vertices using a new histogram from data."""
        self.hist, bins = np.histogram(data, self.bins, density=self.density)
        self.top = self.bottom + self.hist
        self.verts[1::5, 1] = self.top
        self.verts[2::5, 1] = self.top
