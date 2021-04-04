import casadi as ca
import numpy as np
import time
from scipy import interpolate

# To speed up for loops
import multiprocessing
from joblib import Parallel, delayed

class Gridder(object):
    
    def __init__(self, model, Q , R,  
                sim_time = 5):
        """
        A class to solve Dynamic Programming optimization via gridding

        :param model: system model
        :type model: SimplePendulum 
        :param Q: state weight matrix, 2x2
        :type Q: numpy or casadi.DM
        :param R: control weight
        :type R: float
        :param sim_time: simulation time, defaults to 5
        :type sim_time: int, optional
        """
        
        # Initialize class
        self.Q = Q
        self.R = R
        self.sim_time = int(sim_time)
        self.model = model
        self.dt = self.model.dt
        self.DEBUG = False

        # Logged variables
        self.total_cost = 0
        self.total_energy = 0

        # State bounds
        self.x_ub = np.array([np.pi/2, np.pi/4])
        self.x_lb = np.array([-np.pi/2, -np.pi/4])

        # Flags
        self.Jtogo_updated = False
        self.use_nonlinear_controller = False

        self.create_grid()
        
        pass

    def create_grid(self):
        """
        Creates grid for the gridding algorithm.
        """

        eps = np.pi/10
        self.x1 = np.linspace(-np.pi/2+eps, np.pi/2-eps, 10 )
        self.x2 = np.linspace(-np.pi/2, np.pi/2, 5 )

        self.U = np.zeros((len(self.x1), len(self.x2), int(self.sim_time/self.dt) ))
        self.J = np.zeros((len(self.x1), len(self.x2), int(self.sim_time/self.dt) ))

        pass


    def set_cost_functions(self):
        """
        Sets the cost functions expressions.
        """

        x = ca.MX.sym('x', 2)
        u = ca.MX.sym('u', 1)
        Q = ca.MX.sym('Q', (2,2))
        R = ca.MX.sym('R', 1)

        if self.use_nonlinear_controller is False:
            self.Jstage = ca.Function('Jstage', [x, Q, u, R], \
                                      [x.T @ Q @ x + u.T @ R @ u] )
        else:
            # Set nonlinear cost function
            self.Jstage = None 

        self.Jtogo = ca.Function('Jtogo', [x], \
                                  [x.T @ self.Q @ x] )
        pass

    def set_weights(self, Q, R):
        """Set weight matrices Q and R

        :param Q: state weight matrix, 2x2
        :type Q: np.array or ca.diag
        :param R: control weight
        :type R: float
        """
        self.Q = Q
        self.R = R

    def set_linear_controller(self):
        """
        Helper method to set the linear controller functions and grid.
        """
        self.use_nonlinear_controller = False
        self.set_cost_functions()
        self.solve_grid()

    def set_nonlinear_controller(self):
        """
        Helper method to set the nonlinear controller functions and grid.
        """
        self.use_nonlinear_controller = True
        self.set_cost_functions()
        self.solve_grid()

    def solve_grid(self):
        """
        Implements the gridding algorithm by solving the optimization problem 
        for each point in the grid.
        """
        print("Solving grid...")
        
        for n in range(int(self.sim_time/self.dt)):
            print("Time-step: ", "{:.2f}".format(n*self.dt), "/", self.sim_time )
            
            # Get cost at grid point
            for i in range(len(self.x1)):
                for j in range(len(self.x2)):
                    
                    x = np.array([self.x1[i],self.x2[j]])
                    
                    u, cost = self.solve_dp(x)
                    
                    if self.use_nonlinear_controller is False:
                        self.U[i,j,n] = u
                    else:
                        # Modify grid to account for virtual input v
                        self.U[i,j,n] = None
                    
                    self.J[i,j,n] = cost
            
            # Update cost to go
            if self.DEBUG:
                print("U[:,:,", n, "]:\n ", self.U[:,:,n])
                print("J[:,:,", n, "]:\n ", self.J[:,:,n])
            J_flat = self.J[:,:,n]
            J_flat = J_flat.ravel(order='F')

            # Update cost-to-go policy
            self.Jtogo = ca.interpolant('new_cost','bspline', [self.x1, self.x2], J_flat, {"algorithm": "smooth_linear"})
            self.Jtogo_updated = True

        print("Finished!")

        
        pass

    def solve_dp(self, np_x):
        """
        Solves a dynamic programming minimization step.

        :param np_x: current state
        :type np_x: np.array
        :return: optimal control, optimal cost
        :rtype: float, float
        """

        # Set variables as CasADi vars
        x = ca.DM.zeros(2,1)
        x[0,0] = np_x[0]
        x[1,0] = np_x[1]
        u = ca.MX.sym('u', 1,1)

        # Set constraints
        con_ineq, con_ineq_ub, con_ineq_lb = [], [], []

        # Bound on u - you can play with this bound and analyze the effect
        con_ineq.append(u)
        con_ineq_ub.append(10)
        con_ineq_lb.append(-10)

        # Bound on the state constraint
        x_next = self.model.discrete_time_dynamics(x,u)
        con_ineq.append(x_next)
        con_ineq_ub.append(self.x_ub)
        con_ineq_lb.append(self.x_lb)
        
        # Set NLP: X is the optimization variable(s);
        #          F the cost function; 
        #          G is the constraint vector
        # more on: https://web.casadi.org/python-api/#nlp
        if self.Jtogo_updated == False:
            nlp = {'x':u, 'f':self.Jstage(x,self.Q,u,self.R) + \
                    self.Jtogo(x_next), \
                    'g':ca.vertcat(*con_ineq)}
        else:
            nlp = {'x':u, 'f':self.Jstage(x,self.Q,u,self.R) + self.Jtogo(x_next.T), \
                    'g':ca.vertcat(*con_ineq)}

        # Hide solver prints - you can enable them for extra fun !
        opts = {"ipopt.print_level": 0, 
                "print_time": 0}

        # Run solver
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)
        sol = solver(ubg=ca.vertcat(*con_ineq_ub), lbg=ca.vertcat(*con_ineq_lb))

        if self.DEBUG:
            print("Sol: ", sol["x"], " - Cost: ", sol["f"])

        return sol["x"], sol["f"]

    def finite_time_dp(self, x, t):
        """
        Retrieves the optimal control for a given state and time-step.

        :param x: current state
        :type x: numpy array
        :param t: time step
        :type t: float
        :return: optimal control input
        :rtype: float
        """

        n = int(self.sim_time/self.dt) - int(t/self.dt)-1

        self.sample_u = interpolate.RectBivariateSpline(self.x1, self.x2, self.U[:,:,n])
        u = self.sample_u(x=np.array([x[0][0]]), y=np.array([x[1][0]]))

        # Log cost and energy
        self.log_cost(x,u)
        return u

    
    def reset_energy_cost(self):
        """
        Resets the energy and cost for this class.
        """
        self.total_cost = 0
        self.total_energy = 0
        

    def log_cost(self, x, u):
        """
        Log the cost and energy used during the simulation run.

        :param x: state
        :type x: np.array or ca.DM
        :param u: control input
        :type u: float
        """
        self.total_cost = self.total_cost + self.Jstage(x,self.Q,u,self.R)
        self.total_energy = self.total_energy + u.T @ u

    def get_energy_cost(self):
        """
        Returns the total energy and cost of the whole run.

        :return: total energy and cost for the simulation run
        :rtype: float, float
        """
        return self.total_energy, self.total_cost




class BiggerGridder(object):
    
    def __init__(self, model, Q , R,  
                sim_time = 5):
        """
        A class to solve Dynamic Programming optimization via gridding

        :param model: system model
        :type model: PendulumOnCart
        :param Q: state weight matrix, 4x4
        :type Q: numpy or casadi.DM
        :param R: control weight
        :type R: float
        :param sim_time: simulation time, defaults to 5
        :type sim_time: int, optional
        """

        # Initialize class
        self.Q = Q
        self.R = R
        self.sim_time = int(sim_time)
        self.model = model
        self.dt = self.model.dt
        self.DEBUG = False

        # Logged variables
        self.total_cost = 0
        self.total_energy = 0

        # Set solver bounds
        eps = np.pi/20
        self.x_ub = [12, 4, np.pi/2-eps, np.pi/4-eps]
        self.x_lb = [-2, -4, -np.pi/2+eps, -np.pi/4+eps]

        # Flags
        self.Jtogo_updated = False

        self.create_grid()
        
        pass

    def create_grid(self):
        """
        Creates grid for the gridding algorithm.
        """
        # Create state-space grid and fill epsilon variables if needed
        # Hint: take inspiration from Gridder class
        eps = None
        self.x1 = np.linspace()
        self.x2 = np.linspace()

        eps = None
        self.x3 = np.linspace()
        self.x4 = np.linspace()

        self.U = np.zeros((len(self.x1), len(self.x2), 
                            len(self.x3), len(self.x4),
                             int(self.sim_time/self.dt) ))
        self.J = np.zeros((len(self.x1), len(self.x2), 
                            len(self.x3), len(self.x4),
                             int(self.sim_time/self.dt) ))

        pass

    def set_weights(self, Q, R):
        """Set weight matrices Q and R

        :param Q: state weight matrix, 2x2
        :type Q: np.array or ca.diag
        :param R: control weight
        :type R: float
        """

        self.Q = Q
        self.R = R

    def set_controller(self):
        """
        Set cost functions and solve grid.
        """

        self.set_cost_functions()
        self.solve_grid()

    def set_cost_functions(self):
        """
        Set the cost functions expressions to be used by the solver.
        """

        x = ca.MX.sym('x', 4)
        u = ca.MX.sym('u', 1)
        Q = ca.MX.sym('Q', (4,4))
        R = ca.MX.sym('R', 1)

        # Create cost functions for the tracking problem
        # Hint: take inspiration from Gridder class and from the Assignment PDF

        self.Jstage = None

        self.Jtogo = None

        pass

    def set_reference(self, xd=np.array([[10],[0],[0],[0]])):
        """
        Set system reference

        :param xd: reference state, defaults to np.array([[10],[0],[0],[0]])
        :type xd: np.array, optional
        """
        self.xd = xd
        

    def solve_grid(self):
        """
        Implements the gridding algorithm by solving the optimization problem 
        for each point in the grid.
        """

        print("Solving grid...")
        
        for n in range(int(self.sim_time/self.dt)):
            print("Iteration: ", n, "/", int(self.sim_time/self.dt ))
            # Get cost at grid point
            # Hint: take inspiration from Gridder class
            # for ...
            #     for ...
            
            # Update cost to go
            J_flat = self.J[:,:,:,:,n]
            J_flat = J_flat.ravel(order='F')
            
            if self.DEBUG:
                print("U[:,:,", n, "]:\n ", self.U[:,:,n])
                print("J[:,:,:,:,", n, "]:\n ", J_flat)

            # Update cost-to-go policy
            self.Jtogo = ca.interpolant('new_cost','bspline', [self.x1, self.x2, self.x3, self.x4], J_flat, {"algorithm": "smooth_linear"})
            self.Jtogo_updated = True

        print("Finished!")

        
        pass

    def solve_dp(self, np_x):
        """
        Solves a dynamic programming minimization step.

        :param np_x: current state
        :type np_x: np.array
        :return: optimal control, optimal cost
        :rtype: float, float
        """

        # Set variables as CasADi vars
        x = ca.DM.zeros(4,1)
        x[0,0] = np_x[0]
        x[1,0] = np_x[1]
        x[2,0] = np_x[2]
        x[3,0] = np_x[3]
        u = ca.MX.sym('u', 1,1)
        
        # Create bounds lists
        con_ineq, con_ineq_ub, con_ineq_lb = [], [], []

        # Set state bounds
        x_next = self.model.discrete_time_dynamics(x,u)
        con_ineq.append(x_next)
        con_ineq_ub = con_ineq_ub + self.x_ub
        con_ineq_lb = con_ineq_lb + self.x_lb
        
        # Set NLP: X is the optimization variable(s);
        #          F the cost function; 
        #          G is the constraint vector
        # more on: https://web.casadi.org/python-api/#nlp
        if self.Jtogo_updated == False:
            nlp = {'x':u, 'f':self.Jstage(x,self.Q,u,self.R) + \
                    self.Jtogo(x_next), \
                    'g':ca.vertcat(*con_ineq)}
        else:
            nlp = {'x':u, 'f':self.Jstage(x,self.Q,u,self.R) + self.Jtogo(x_next.T), \
                    'g':ca.vertcat(*con_ineq)}

        # Solver options are blank by default: do a careful analysis of its
        # output for debugging problems with your grid or starting conditions
        # Hint: to disable outputs, you can take the options in the Gridder class
        opts = {} 


        # Run solver
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)
        sol = solver(ubg=ca.vertcat(*con_ineq_ub), lbg=ca.vertcat(*con_ineq_lb))

        if self.DEBUG:
            next_state = self.model.discrete_time_dynamics(x,sol["x"])
            print("Initial state: ", x)
            print("Next state: ", next_state)
            print("U: ", sol["x"])
            violation = False
            if abs(next_state[0]) > self.x_ub[0]: 
                violation = True
                print("Violated position bound: ", self.x_ub[0] )
            
            if abs(next_state[1]) > self.x_ub[1]:
                violation = True
                print("Violated velocity bound: ", self.x_ub[1])

            if abs(next_state[2]) > self.x_ub[2]:
                violation = True
                print("Violated angle bound: ", self.x_ub[2])

            if abs(next_state[3]) > self.x_ub[3]:
               violation = True
               print("Violated angular velocity bound: ", self.x_ub[3])
            
            if violation:
               input()

        return sol["x"], sol["f"]


    def finite_time_dp(self, x, t):
        """
        Retrieves the optimal control for a given state and time-step.

        :param x: current state
        :type x: numpy array
        :param t: time step
        :type t: float
        :return: optimal control input
        :rtype: float
        """
    
        n = int(self.sim_time/self.dt) - int(t/self.dt)-1

        U_flat = self.U[:,:,:,:,n].ravel(order="F")
        self.sample_u = ca.interpolant('new_cost','bspline', 
                                        [self.x1, self.x2, self.x3, self.x4], 
                                        U_flat, {"algorithm": "smooth_linear"})

        u = self.sample_u(x.T)
        
        # Log cost and energy
        self.log_cost(x,u)
        return u

    
    def reset_energy_cost(self):
        """
        Reset stored energy and cost for the simulation run.

        :param x: [description]
        :type x: [type]
        :param u: [description]
        :type u: [type]
        """
        
        self.total_cost = 0
        self.total_energy = 0
        pass

    def log_cost(self, x, u):
        """
        Log the energy and cost used during the simulation run.

        :param x: state
        :type x: np.array
        :param u: control input
        :type u: float
        """
        self.total_cost = self.total_cost + self.Jstage(x,self.Q,u,self.R)
        self.total_energy = self.total_energy + u.T @ u

    def get_energy_cost(self):
        """
        Returns the total energy and cost of a simulation run.

        :return: total energy and total cost
        :rtype: float, float
        """
        return self.total_energy, self.total_cost