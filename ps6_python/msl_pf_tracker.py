import cv2
import numpy as np
from compare_hist import *

class MSLPFTracker:
    '''
    A Particle Filter Tracker using Mean-Shift-Lite
    '''
    def __init__(self, model, search_space, num_particles=100, state_dims=2,
                 control_std=10, sim_std=20, alpha=0.0):
        self.model = model
        self.search_space = search_space[::-1]
        self.num_particles = num_particles
        self.state_dims = state_dims
        self.control_std = control_std
        self.sim_std = sim_std
        self.alpha = alpha
        # inialize particles using a uniform distribution
        self.particles = np.array([np.random.uniform(0, self.search_space[i],
                                                     self.num_particles)
                                   for i in range(self.state_dims)]).T
        self.weights = np.ones(len(self.particles)) / len(self.particles)
        self.idxs = np.arange(num_particles)
        self.estimate_state()

    def update(self, frame):
        self.displace()
        self.observe(frame)
        self.resample()
        self.estimate_state()
        if self.alpha > 0:
            self.update_model(frame)

    def displace(self):
        # displace particles using a normal distribution centered around 0
        self.particles += np.random.normal(0, self.control_std,
                                           self.particles.shape)

    def observe(self, img):
        # get patches corresponding to each particle
        mh, mw = self.model.shape[:2]
        minx = (self.particles[:,0] - mw/2).astype(np.int)
        miny = (self.particles[:,1] - mh/2).astype(np.int)
        candidates = [img[miny[i]:miny[i]+mh, minx[i]:minx[i]+mw]
                      for i in range(self.num_particles)]
        # compute importance weight - similarity of each patch to the model
        self.weights[:] = [compare_hist(cand, self.model, std=10,
                                              num_bins=8)
                                 for cand in candidates]
        # normalize the weights
        self.weights /= self.weights.sum()

    def resample(self):
        sw, sh = self.search_space[:2]
        mh, mw = self.model.shape[:2]
        # sample new particle indices using the distribution of the weights
        distrib = self.weights.flatten()
        j = np.random.choice(self.idxs, self.num_particles, True, p=distrib)
        # get a random control input from a normal distribution
        control = np.random.normal(0, self.control_std, self.particles.shape)
        # sample the particles using the distribution of the weights
        self.particles = np.array(self.particles[j])
        # clip particles in case the window goes out of the image limits
        self.particles[:,0] = np.clip(self.particles[:,0], 0, sw - 1)
        self.particles[:,1] = np.clip(self.particles[:,1], 0, sh - 1)

    def estimate_state(self):
        state_idx = np.random.choice(self.idxs, 1, p=self.weights.flatten())
        self.state = self.particles[state_idx][0]

    def update_model(self, frame):
        # get current model based on belief
        mh, mw = self.model.shape[:2]
        minx = int(self.state[0] - mw/2)
        miny = int(self.state[1] - mh/2)
        best_model = frame[miny:miny+mh, minx:minx+mw]
        # apply appearance model update if new model shape is unchanged
        if best_model.shape == self.model.shape:
            self.model = self.alpha * best_model + (1-self.alpha) * self.model
            self.model = self.model.astype(np.uint8)

    def visualize_filter(self, img):
        self.draw_particles(img)
        self.draw_window(img)
        self.draw_std(img)

    def draw_particles(self, img):
        for p in self.particles:
            cv2.circle(img, tuple(p.astype(int)), 2, (180,255,0), -1)

    def draw_window(self, img):
        best_idx = cv2.minMaxLoc(self.weights)[3][1]
        best_state = self.particles[best_idx]
        pt1 = (best_state - np.array(self.model.shape[1::-1])/2).astype(np.int)
        #  pt1 = (self.state - np.array(self.model.shape[1::-1])/2).astype(np.int)
        pt2 = pt1 + np.array(self.model.shape[1::-1])
        cv2.rectangle(img, tuple(pt1), tuple(pt2), (0,255,0), 2)

    def draw_std(self, img):
        weighted_sum = 0
        dist = np.linalg.norm(self.particles - self.state)
        weighted_sum = np.sum(dist * self.weights.ravel())
        cv2.circle(img, tuple(self.state.astype(np.int)),
                   int(weighted_sum), (255,255,255), 1)

