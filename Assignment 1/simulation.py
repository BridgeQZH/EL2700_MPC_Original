import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import time

class EmbeddedSimEnvironment(object):
    
    def __init__(self, model, dynamics, controller, time=100.0):
        """
        Embedded simulation environment. Simulates the syste given dynamics 
        and a control law, plots in matplotlib.

        :param model: model object
        :type model: object
        :param dynamics: system dynamics function (x, u)
        :type dynamics: casadi.DM
        :param controller: controller function (x, r)
        :type controller: casadi.DM
        :param time: total simulation time, defaults to 100 seconds
        :type time: float, optional
        """
        self.model = model
        self.dynamics = dynamics
        self.controller = controller
        self.total_sim_time = time # seconds
        self.dt = self.model.dt

        # Plotting definitions 
        self.plt_window = float("inf")    # running plot window, in seconds, or float("inf")

    def run(self, x0=[0,0,0,0]):
        """
        Run simulator with specified system dynamics and control function.
        """
        
        print("Running simulation....")
        sim_loop_length = int(self.total_sim_time/self.dt) + 1 # account for 0th
        t = np.array([0])
        y_vec = np.array([x0]).T
        u_vec = np.array([0])
        
        # Start figure
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        for i in range(sim_loop_length):
            
            # Translate data to ca.DM
            x = ca.DM(4,1).full()
            x = y_vec[:,-1]

            try:
                # Get control input and obtain next state
                u = self.controller(x)
                x_next = self.dynamics(x, u)
            except RuntimeError:
                print("Uh oh, your simulator crashed due to unstable dynamics.\n \
                       Retry with new controller parameters.")
                exit()

            # Store data
            t = np.append(t,t[-1]+self.dt)
            y_vec = np.append(y_vec, np.array(x_next), axis=1)
            u_vec = np.append(u_vec, np.array(u))

            # Get plot window values:
            if self.plt_window != float("inf"):
                l_wnd = 0 if int(i+1 - self.plt_window/self.dt) < 1 else int(i+1 - self.plt_window/self.dt)
            else:  
                l_wnd = 0

            ax1.clear()
            plt.subplot(211)
            plt.plot( t[l_wnd:-1], y_vec[0,l_wnd:-1], 'r--', \
                      t[l_wnd:-1], y_vec[1,l_wnd:-1], 'b--')
            plt.legend(["x1","x2"])
            plt.ylabel("X1 [m] / X2 [m/s]")
            plt.title("Pendulum on Cart - Ref: "+str(self.model.x_d)+" [m]")

            #ax2.clear()
            plt.subplot(212)
            plt.plot( t[l_wnd:-1], y_vec[2,l_wnd:-1], 'g--', \
                      t[l_wnd:-1], y_vec[3,l_wnd:-1], 'k--')
            plt.xlabel("Time [s]")
            plt.ylabel("X3 [rad] / X4 [rad/s]")
            plt.legend(["x3","x4"])
            plt.pause(0.01)
            
        plt.show()
        return t, y_vec, u_vec

        

