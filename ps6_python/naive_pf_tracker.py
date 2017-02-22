import cv2
import numpy as np
from similarity import *

class NaivePFTracker:
    def __init__(self, model, search_space, num_particles, state_dims,
                 control_scale, control_noise, sim_std):
        self.model = model
        self.search_space = search_space
        self.num_particles = num_particles
        self.state_dims = state_dims
        self.control_scale = control_scale
        self.control_noise = control_noise
        self.sim_std = sim_std
        # initialize particles
        self.particles = np.zeros((num_particles, state_dims))
        self.particles[:, 0] = np.random.uniform(0, search_space[0],
                                                 num_particles)
        self.particles[:, 1] = np.random.uniform(0, search_space[1],
                                                num_particles)
        self.weights = np.ones(len(self.particles)) / len(self.particles)
        self.state = (self.particles * self.weights.reshape((-1, 1))).sum(0)
        self.idxs = np.arange(num_particles)

    def resample(self, img):  # generate new samples
        new_particles = np.zeros(self.particles.shape)
        new_weights = np.zeros(self.weights.shape)
        sh, sw = self.search_space
        mh, mw = self.model.shape[:2]


        # sample index j[i] from w_t-1
        j = np.random.choice(self.idxs, self.num_particles, p=self.weights.T)
        # get a random control input
        control = np.random.normal(0, self.control_scale, self.state_dims)
        # add noise to the control input of all particles
        noisy_control = control + np.random.normal(0, self.control_noise,
                                                   (self.num_particles, 2))
        # sample x_t^i from p(x_t | x_t-1, u_t) using x_t^j and u_t
        new_particles = np.array(self.particles[j] + noisy_control)
        # clip particles in case the window goes out of the image limits
        new_particles[:,0] = np.clip(new_particles[:,0], mh/2, sh - mh/2 - 1)
        new_particles[:,1] = np.clip(new_particles[:,1], mw/2, sw - mw/2 - 1)
        # get patches corresponding to particle i
        miny = (new_particles[:,0] - mh/2).astype(np.int); maxy = miny + mh
        minx = (new_particles[:,1] - mw/2).astype(np.int); maxx = minx + mw
        candidates = [img[miny[i]:maxy[i], minx[i]:maxx[i]]
                      for i in range(self.num_particles)]
        # compute importance weight - similarity of each patch to the model
        new_weights = np.array([similarity(cand, self.model, self.sim_std)
                               for cand in candidates])

        # normalize new weights
        new_weights /= np.sum(new_weights)

        # store new particles and weights and discard the old
        self.particles = new_particles
        self.weights = new_weights

        # estimate current state - weighted mean of particle states
        self.state = (self.particles * self.weights.reshape((-1, 1))).sum(0)

    def visualize_filter(self, img):
        self.draw_particles(img)
        self.draw_window(img)
        self.draw_std(img)

    def draw_particles(self, img):
        for p in self.particles:
            cv2.circle(img, tuple(p[::-1].astype(int)), 2, (180,255,0), -1)

    def draw_window(self, img):
        pt1 = (self.state - np.array(self.model.shape)/2).astype(np.int)
        pt2 = pt1 + np.array(self.model.shape)
        cv2.rectangle(img, tuple(pt1[::-1]), tuple(pt2[::-1]), (0,255,0), 2)

    def draw_std(self, img):
        weighted_sum = 0
        dist = np.linalg.norm(self.particles - self.state)
        weighted_sum = np.sum(dist * self.weights.reshape((-1,1)))
        cv2.circle(img, tuple(self.state[::-1].astype(np.int)),
                   int(weighted_sum), (255,255,255), 1)

