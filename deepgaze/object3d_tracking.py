#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2016 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#Thank to rlabbe and his fantastic repository for Bayesian Filter:
#https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

import math
import numpy as np
from numpy.random import uniform
import cv2
import sys

class ParticleFilter3D:
    """Particle filter motion tracking.

    This class estimates the position of a single point in a 3d space.
    It can be used to predict the position of a landmark for example when tracking some face features, or
    to track the corner of a bounding box.
    """

    # def __init__(self, width, height, depth, N):
    def __init__(self, width, height, N):
        """Init the particle filter.

        @param width the width of the bounding box
        @param height the height of the bounding box
        @param deptht the depth of the bounding box
        @param N the number of particles
        """
        # if(N <= 0 or N>width*height): 
        if(N <= 0):
            raise ValueError('[DEEPGAZE] motion_tracking.py: the ParticleFilter class does not accept a value of N which is <= 0 or >(widht*height)')
        self.particles = np.empty((N, 2))
        self.particles[:, 0] = uniform(-0.5*width, 0.5*width, size=N) #init the X coord
        self.particles[:, 1] = uniform(-0.5*height, 0.5*height, size=N) #init the Y coord
        # self.particles[:, 2] = uniform(-0.5*depth, 0.5*depth, size=N) #init the Y coord
        #Init the weiths vector as a uniform distribution
        #at the begining each particle has the same probability
        #to represent the point we are following
        #self.weights = np.empty((N, 1))
        self.weights = np.array([1.0/N]*N)
        #self.weights.fill(1.0/N) #normalised values
        self.context_x=-1.2
        self.context_y=-1

        self.context_x2=1.7
        self.context_y2=-1
    # def predict(self, x_velocity, y_velocity, z_velocity, std ):
    def predict(self, x_velocity, y_velocity, std ):
        """Predict the position of the point in the next frame.
        Move the particles based on how the real system is predicted to behave.
 
        The position of the point at the next time step is predicted using the 
        estimated velocity along X and Y axis and adding Gaussian noise sampled 
        from a distribution with MEAN=0.0 and STD=std. It is a linear model.
        @param x_velocity the velocity of the object along the X axis in terms of pixels/frame
        @param y_velocity the velocity of the object along the Y axis in terms of pixels/frame
        @param z_velocity the velocity of the object along the Z axis in terms of pixels/frame
        @param std the standard deviation of the gaussian distribution used to add noise
        """
        #To predict the position of the point at the next step we take the
        #previous position and we add the estimated speed and Gaussian noise
        self.particles[:, 0] += uniform(-1.0,1.0)*x_velocity + (np.random.randn(len(self.particles)) * std) #predict the X coord
        self.particles[:, 1] += uniform(-1.0,1.0)*y_velocity + (np.random.randn(len(self.particles)) * std) #predict the Y coord
        # self.particles[:, 2] += z_velocity + (np.random.randn(len(self.particles)) * std) #predict the Y coord
    def predict_context(self,context_category,std ):
        """Predict the position of the point in the next frame from context information.
        Move the particles based on how the real system is predicted to behave.
        """
        # print context_category
        if context_category==1:
            for i in range(len(self.particles)):
                if i<0.5*len(self.particles):
                    # self.particles[i, 0] = self.context_x + (np.random.randn(len(self.particles)) * std) #predict the X coord
                    self.particles[i, 0] = self.context_x + 5*(np.random.uniform(-1.0, 1.0) * std) #predict the X coord
                    self.particles[i, 1] = self.context_y + 5*(np.random.uniform(-1.0, 1.0) * std) #predict the Y coord
                    # self.particles[i, 0] = self.context_x + (np.random.randn(len(self.particles)) * std) #predict the X coord
                    # self.particles[i, 1] = self.context_y + (np.random.randn(len(self.particles)) * std) #predict the Y coord
                    # self.particles[i, 1] = self.context_y + (np.random.randn(len(self.particles)) * std) #predict the Y coord
                else:
                    self.particles[i, 0] = self.context_x2 + 5*(np.random.uniform(-1.0, 1.0) * std) #predict the X coord
                    self.particles[i, 1] = self.context_y2 + 5*(np.random.uniform(-1.0, 1.0) * std) #predict the Y coord
        else:
            self.particles[:, 0] = -self.context_x + (np.random.randn(len(self.particles)) * std) #predict the X coord
            self.particles[:, 1] = -self.context_y + (np.random.randn(len(self.particles)) * std) #predict the Y coord
            
        # print "predict context"
        # self.particles[:, 2] += z_velocity + (np.random.randn(len(self.particles)) * std) #predict the Y coord


    # def update(self, x, y,z):
    def update(self, x, y):
        """Update the weights associated which each particle based on the (x,y,z) coords measured.
        Particles that closely match the measurements give an higher contribution.
 
        The position of the point at the next time step is predicted using the 
        estimated speed along X and Y axis and adding Gaussian noise sampled 
        from a distribution with MEAN=0.0 and STD=std. It is a linear model.
        @param x the position of the point in the X axis
        @param y the position of the point in the Y axis
        @param 
        """
        #Generating a temporary array for the input position
        # position = np.empty((len(self.particles), 3))
        position = np.empty((len(self.particles), 2))
        position[:, 0].fill(x)
        position[:, 1].fill(y)
        # position[:, 2].fill(z)
        #1- We can take the difference between each particle new
        #position and the measurement. In this case is the Euclidean Distance.
        distance = np.linalg.norm(self.particles - position, axis=1)
        #2- Particles which are closer to the real position have smaller
        #Euclidean Distance, here we subtract the maximum distance in order
        #to get the opposite (particles close to the real position have
        #an higher wieght)
        max_distance = np.amax(distance)
        distance = np.add(-distance, max_distance)
        #3-Particles that best predict the measurement 
        #end up with the highest weight.
        self.weights.fill(1.0) #reset the weight array
        self.weights *= distance
        #4- after the multiplication the sum of the weights won't be 1. 
        #Renormalize by dividing all the weights by the sum of all the weights.
        self.weights += 1.e-300 #avoid zeros
        self.weights /= sum(self.weights) #normalize
        # print "----------predit-----------" 
        # print  self.weights
        entropy = self.get_entropy()
        return entropy

    def update_fake(self, sub_particles, x, y):
        """Update the weights associated which each particle based on the (x,y,z) coords measured.
        Particles that closely match the measurements give an higher contribution.
 
        The position of the point at the next time step is predicted using the 
        estimated speed along X and Y axis and adding Gaussian noise sampled 
        from a distribution with MEAN=0.0 and STD=std. It is a linear model.
        @param x the position of the point in the X axis
        @param y the position of the point in the Y axis
        @param 
        """
        #Generating a temporary array for the input position
        # position = np.empty((len(self.particles), 3))
        if len(sub_particles)<50:
            print "not enough particles to evaluate"

        N_subparticles=len(sub_particles)
        position = np.empty((len(sub_particles), 2))
        position[:, 0].fill(x)
        position[:, 1].fill(y)
        # position[:, 2].fill(z)
        #1- We can take the difference between each particle new
        #position and the measurement. In this case is the Euclidean Distance.
        distance = np.linalg.norm(sub_particles- position, axis=1)
        #2- Particles which are closer to the real position have smaller
        #Euclidean Distance, here we subtract the maximum distance in order
        #to get the opposite (particles close to the real position have
        #an higher wieght)
        max_distance = np.amax(distance)
        distance = np.add(-distance, max_distance)
        #3-Particles that best predict the measurement 
        #end up with the highest weight.
        fake_weights = np.empty((N_subparticles, 1))
        fake_weights.fill(1.0) #reset the weight array
        fake_weights *= distance
        #4- after the multiplication the sum of the fake_weights won't be 1. 
        #Renormalize by dividing all the fake_weights by the sum of all the fake_weights.
        fake_weights += 1.e-300 #avoid zeros
        fake_weights /= sum(fake_weights) #normalize
        # print "----------predit-----------" 
        # print  self.fake_weights
        # entropy = self.get_entropy()
        return fake_weights



    def get_entropy_sample(self, x, y):
        """Update the weights associated which each particle based on the (x,y,z) coords measured.
        Particles that closely match the measurements give an higher contribution.
 
        The position of the point at the next time step is predicted using the 
        estimated speed along X and Y axis and adding Gaussian noise sampled 
        from a distribution with MEAN=0.0 and STD=std. It is a linear model.
        @param x the position of the point in the X axis
        @param y the position of the point in the Y axis
        @param 
        """
        #Generating a temporary array for the input position
        # position = np.empty((len(self.particles), 3))
        position = np.empty((len(self.particles), 2))
        position[:, 0].fill(x)
        position[:, 1].fill(y)
        # position[:, 2].fill(z)
        #1- We can take the difference between each particle new
        #position and the measurement. In this case is the Euclidean Distance.
        distance = np.linalg.norm(self.particles - position, axis=1)
        #2- Particles which are closer to the real position have smaller
        #Euclidean Distance, here we subtract the maximum distance in order
        #to get the opposite (particles close to the real position have
        #an higher wieght)
        max_distance = np.amax(distance)
        distance = np.add(-distance, max_distance)
        #3-Particles that best predict the measurement 
        #end up with the highest weight.
        sampledweights.fill(1.0) #reset the weight array
        sampledweights*= distance
        #4- after the multiplication the sum of the weights won't be 1. 
        #Renormalize by dividing all the weights by the sum of all the weights.
        sampledweights += 1.e-300 #avoid zeros
        sampledweights /= sum(sampledweights) #normalize
        # print "----------predit-----------" 
        # print  self.weights
        sample_entropy =-np.sum(sampledweights*np.log2(sampledweights))
        return sample_entropy



    def estimate(self):
        """Estimate the position of the point given the particle weights.
 
        This function get the mean and variance associated with the point to estimate.
        @return get the x_mean, y_mean and the x_var, y_var 
        """
        #Using the weighted average of the particles
        #gives an estimation of the position of the point
        x_mean = np.average(self.particles[:, 0], weights=self.weights, axis=0).astype(float)
        y_mean = np.average(self.particles[:, 1], weights=self.weights, axis=0).astype(float)
        # z_mean = np.average(self.particles[:, 2], weights=self.weights, axis=0).astype(int)

        #mean = np.average(self.particles[:, 0:3], weights=self.weights, axis=0)
        #var  = np.average((self.particles[:, 0:3] - mean)**2, weights=self.weights, axis=0)
        #x_mean = int(mean[0])
        #y_mean = int(mean[1])
        #z_mean = int(mean[2])
        #x_var = int(var[0])
        #y_var = int(var[1])
        #z_var = int(var[2])
        # return x_mean, y_mean,z_mean, 0, 0, 0
        return x_mean, y_mean, 0, 0

    # def resample(self, method='multinomal'):
    def resample(self, method='residual'):
        """Resample the particle based on their weights.
 
        The resempling (or importance sampling) draws with replacement N
        particles from the current set with a probability given by the current
        weights. The new set generated has always size N, and it is an
        approximation of the posterior distribution which represent the state
        of the particles at time t. The new set will have many duplicates 
        corresponding to the particles with highest weight. The resampling
        solve a huge problem: after some iterations of the algorithm
        some particles are useless because they do not represent the point 
        position anymore, eventually they will be too far away from the real position.
        The resample function removes useless particles and keep the
        useful ones. It is not necessary to resample at every epoch.
        If there are not new measurements then there is not any information 
        from which the resample can benefit. To determine when to resample 
        it can be used the returnParticlesContribution function.
        @param method the algorithm to use for the resampling.
            'multinomal' large weights are more likely to be selected [complexity O(n*log(n))]
            'residual' (default value) it ensures that the sampling is uniform across particles [complexity O(N)]
            'stratified' it divides the cumulative sum into N equal subsets, and then 
                selects one particle randomly from each subset.
            'systematic' it divides the cumsum into N subsets, then add a random offset to all the susets
        """
        N = len(self.particles)
        if(method == 'multinomal'):
            #np.cumsum() computes the cumulative sum of an array. 
            #Element one is the sum of elements zero and one, 
            #element two is the sum of elements zero, one and two, etc.
            cumulative_sum = np.cumsum(self.weights)
            cumulative_sum[-1] = 1. #avoid round-off error
            #np.searchsorted() Find indices where elements should be 
            #inserted to maintain order. Here we generate random numbers 
            #in the range [0.0, 1.0] and do a search to find the weight 
            #that most closely matches that number. Large weights occupy 
            #more space than low weights, so they will be more likely 
            #to be selected.
            indices = np.searchsorted(cumulative_sum, np.random.uniform(low=0.0, high=1.0, size=N))      
        elif(method == 'residual'):
            indices = np.zeros(N, dtype=np.int32)
            # take int(N*w) copies of each weight
            num_copies = (N*np.asarray(self.weights)).astype(int)
            k = 0
            for i in range(N):
                for _ in range(num_copies[i]): # make n copies
                    indices[k] = i
                    k += 1
            #multinormial resample
            residual = self.weights - num_copies     # get fractional part
            residual /= sum(residual)     # normalize
            cumulative_sum = np.cumsum(residual)
            cumulative_sum[-1] = 1. # ensures sum is exactly one
            indices[k:N] = np.searchsorted(cumulative_sum, np.random.random(N-k))
        elif(method == 'stratified'):
            #N subsets, chose a random position within each one
            #and generate a vector containing this positions
            positions = (np.random.random(N) + range(N)) / N
            #generate the empty indices vector
            indices = np.zeros(N, dtype=np.int32)
            #get the cumulative sum
            cumulative_sum = np.cumsum(self.weights)
            i, j = 0, 0
            while i < N:
                if positions[i] < cumulative_sum[j]:
                    indices[i] = j
                    i += 1
                else:
                    j += 1
        elif(method == 'systematic'):
            # make N subsets, choose positions with a random offset
            positions = (np.arange(N) + np.random.random()) / N
            indices = np.zeros(N, dtype=np.int32)
            cumulative_sum = np.cumsum(self.weights)
            i, j = 0, 0
            while i < N:
                if positions[i] < cumulative_sum[j]:
                    indices[i] = j
                    i += 1
                else:
                    j += 1
        else:
            raise ValueError("[DEEPGAZE] motion_tracking.py: the resempling method selected '" + str(method) + "' is not implemented")
        #Create a new set of particles by randomly choosing particles 
        #from the current set according to their weights.
        self.particles[:] = self.particles[indices] #resample according to indices
        self.weights[:] = self.weights[indices]
        #Normalize the new set of particles
        # print "----------resample-----------" 
        # print self.weights
        self.weights /= np.sum(self.weights)        
        # print'weight size',len(self.weights)
    def get_entropy(self):
        self.entropy=0
        # for each in self.weights:
            # self.entropy=self.entropy-each*np.log2(each);
        # self.entropy =-np.sum(self.weights*math.log(self.weights))
        self.entropy =-np.sum(self.weights*np.log2(self.weights))
        return self.entropy



    def returnParticlesContribution(self):
        """This function gives an estimation of the number of particles which are
        contributing to the probability distribution (also called the effective N). 
 
        This function get the effective N value which is a good estimation for
        understanding when it is necessary to call a resampling step. When the particle
        are collapsing in one point only some of them are giving a contribution to 
        the point estimation. If the value is less than N/2 then a resampling step
        should be called. A smaller value means a larger variance for the weights, 
        hence more degeneracy
        @return get the effective N value. 
        """
        return 1.0 / np.sum(np.square(self.weights))

    def returnParticlesCoordinates(self, index=-1):
        """It returns the (x,y) coord of a specific particle or of all particles. 
 
        @param index the position in the particle array to return
            when negative it returns the whole particles array
        @return a single coordinate (x,y) or the entire array
        """
        if(index<0):
            return self.particles.astype(float)
        else:
            return self.particles[index,:].astype(float)

    def drawParticles(self, frame, color=[0,0,255], radius=2):
        """Draw the particles on a frame and return it.
 
        @param frame the image to draw
        @param color the color in BGR format, ex: [0,0,255] (red)
        @param radius is the radius of the particles
        @return the frame with particles
        """
        for x_particle, y_particle in self.particles.astype(int):
            cv2.circle(frame, (x_particle, y_particle), radius, color, -1) #RED: Particles


