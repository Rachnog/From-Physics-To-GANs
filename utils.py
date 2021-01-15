import numpy as np
from math import *

def pendulum_generation(
                            theta_0,
                            omega_0,
                            ti,
                            m,
                            g,
                            l,
                            N
                       ):
    '''
        This method generates trajectories of a simple pendulum motion using the Euler-Cromer method, 
        see https://jeremykao.net/2017/07/03/simulating-a-simple-pendulum-in-python-part-1/ for more details
        
        @theta_0: the initial angle, in radians
        @omega_0: the initial angular velocity, in radians / second
        @ti: the time step
        @m: mass of the pendulum
        @g: gravitational constant
        @l: length of the pendulum
        @N: number of time steps
        
        returns: vectors with time and corresponding angles
    '''
    
    # arrays to return
    time_steps = np.zeros(N)
    theta_steps = np.zeros(N)
    omega_steps = np.zeros(N)

    # setting up initial conditions
    theta = theta_0
    omega = omega_0
    time = 0

    for i in range(0, N):
        # looping, solving the equation numerically
        omega_i = omega
        theta_i = theta
        omega = omega_i - (g/l)*sin(theta_i)*ti
        theta = theta_i + omega*ti

        time_steps[i] = ti*i
        theta_steps[i] = theta
        omega_steps[i] = omega
    
    return time_steps, theta_steps

def approximation_dataset_generator(
                        theta_0_range,
                        omega_0_range,
                        m_range,
                        l_range,
                        g = 10,
                        ti = 0.01,
                        N = 3000
                    ):
    
    '''
        Creates a dataset for a "fake" generative model that generates a trajectory
        step by step from the degrees of freedom and time steps
    
        @theta_0_range: range of values of theta for training
        @omega_0_range: range of values of omega for training
        @ti: the time step
        @m: mass of the pendulum
        @g: gravitational constant
        @l: length of the pendulum
        @N: number of time steps
        
        returns: X and Y, pairs for the training dataset      
    '''
    
    X, Y = [], []
    
    for theta_0_i in theta_0_range:
        for omega_0_i in omega_0_range:
            for m in m_range:
                for l in l_range:
                    
                    theta_0_i += np.random.normal(0, 0.05)
                    omega_0_i += np.random.normal(0, 0.05)
                    m += np.random.normal(0, 0.05)
                    g += np.random.normal(0, 0.05)
                    l += np.random.normal(0, 0.05)
            
                    t, v = pendulum_generation(theta_0_i, omega_0_i, ti, m, g, l, N)
                    for ti_i, vi in zip(t, v):
                        x_i = np.array([theta_0_i, omega_0_i, ti_i, m, g, l])
                        y_i = vi + np.random.normal(0, ti_i * 1e-1)
                        X.append(x_i)
                        Y.append(y_i)
                
    return np.array(X), np.array(Y)


def distribution_dataset_generator(
                        theta_0_range,
                        omega_0_range,
                        m_range,
                        l_range,
                        g = 10,
                        ti = 0.01,
                        N = 3000
                    ):
    
    '''
        Creates a dataset for a "true" data distribution model to be trained as GAN or VAE or similar
    
        @theta_0_range: range of values of theta for training
        @omega_0_range: range of values of omega for training
        @ti: the time step
        @m: mass of the pendulum
        @g: gravitational constant
        @l: length of the pendulum
        @N: number of time steps
        
        returns: X and Y, pairs for the training dataset      
    '''
    
    X = []
    for theta_0_i in theta_0_range:
        for omega_0_i in omega_0_range:
            for m in m_range:
                for l in l_range:
                    
                    theta_0_i += np.random.normal(0, 0.05)
                    omega_0_i += np.random.normal(0, 0.05)
                    m += np.random.normal(0, 0.05)
                    g += np.random.normal(0, 0.05)
                    l += np.random.normal(0, 0.05)
            
                    t, v = pendulum_generation(theta_0_i, omega_0_i, ti, m, g, l, N)
                    xi = []
                    for ti_i, vi in zip(t, v):
                        y_i = vi + np.random.normal(0, ti_i * 1e-1)
                        xi.append(y_i)
                    X.append(xi)
                
    return np.array(X)