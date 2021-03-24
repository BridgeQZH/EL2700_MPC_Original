import casadi as ca

class Pendulum(object):
    def __init__(self, h=):
        """
        Pendulum model class. 
        
        Describes the movement of a pendulum with mass 'm' attached to a cart
        with mass 'M'. All methods should return casadi.MX or casadi.DM variable 
        types.

        :param h: sampling time, defaults to 0.1
        :type h: float, optional
        """

        # Model, gravity and sampling time parameters
        self.model = self.pendulum_linear_dynamics
        self.model_nl = self.pendulum_nonlinear_dynamics
        self.g = 9.81
        self.dt = h

        # System reference (x_d) and disturbance (w)
        self.x_d = 10
        self.w = 0.0

        # Pendulum Parameters
        self.m = 0.2
        self.M = 0.5
        self.I = 0.006
        self.l = 0.3
        self.bc = 0.1
        self.bp = 0.012

        # Linearize system around vertical equilibrium with no input
        self.x_eq = [0,0,0,0]
        self.u_eq = 0
        self.Integrator_lin = None
        self.Integrator = None

        self.set_integrators()
        self.set_discrete_time_system()

        print("Pendulum class initialized")
        print(self)                         # You can comment this line

    def __str__(self):
        return """                                                                  
                ,@@,                                                            
              @@@@@@@@                                                          
              @@@ m @@                                                         
               .@@@@&                                                          
                     *                   .                                      
                      (#       theta     .                                   
                       ,* *              .                                   
                         */              .                                      
                          (*             .                                      
                            *,           .                                      
                             #(          .                                      
           Y               l   *         .                                      
           ^                    ##       .                                      
           |                     .*      .                                      
           |                       /(    .                                      
           |                        /*   .                                      
           +-------> X                *, .                                      
                                       %/.                                      
                        ***************////***************                      
                        (*********************************      F              
                        (***  M  *************************---------->       
                        (*********************************                      
                            ,/**/                ,/*/*                          
                          ********#            (*******(                        
                          #*******(            %*******#                        
                            %***#                %***%                          
            -----------------------------------------------------------      """

    def set_integrators(self):
        """
        Generate continuous time high-precision integrators.
        """
        
        # Set CasADi variables
        x = ca.MX.sym('x', 4)
        u = ca.MX.sym('u', 1)

        # Integration method - integrator options an be adjusted
        options = {"abstol" : 1e-5, "reltol" : 1e-9, "max_num_steps": 100, 
                   "tf" : self.dt}

        # Create linear dynamics integrator
        dae = 
        self.Integrator_lin = 
        
        # Create nonlinear dynamics integrator
        dae_nl = {'x': x, 'ode': self.model_nl(x,u), 'p':ca.vertcat(u)}
        self.Integrator = ca.integrator('integrator', 'cvodes', dae_nl, options)

    def set_discrete_time_system(self):
        """
        Set discrete-time system matrices from linear continuous dynamics.
        """
        
        # Check for integrator definition
        if self.Integrator_lin is None:
            print("Integrator_lin not defined. Set integrators first.")
            exit()

        # Set CasADi variables
        x = ca.MX.sym('x', 4)
        u = ca.MX.sym('u', 1)
    
        # Jacobian of exact discretization
        self.Ad = ca.Function('jac_x_Ad', [x, u], [ca.jacobian(
                            self.Integrator_lin(x0=x, p=u)['xf'], x)])
        self.Bd = ca.Function('jac_u_Bd', [x, u], [ca.jacobian(
                            self.Integrator_lin(x0=x, p=u)['xf'], u)])


    def pendulum_linear_dynamics(self, x, u, *_):  
        """ 
        Pendulum continuous-time linearized dynamics.

        :param x: state
        :type x: MX variable, 4x1
        :param u: control input
        :type u: MX variable, 1x1
        :return: dot(x)
        :rtype: MX variable, 4x1
        """

        Ac = ca.MX.zeros(4,4)
        Bc = ca.MX.zeros(4,1)

        Ac =
        Bc =

        ### Store matrices as class variables
        self.Ac = Ac
        self.Bc = Bc 

        return Ac @ x + Bc @ u  

    def pendulum_nonlinear_dynamics(self, x, u, *_):
        """
        Pendulum nonlinear dynamics.

        :param x: state
        :type x: casadi.DM or casadi.MX
        :param u: control input
        :type u: casadi.DM or casadi.MX
        :return: state time derivative
        :rtype: casadi.DM or casadi.MX, depending on inputs
        """
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
         

        f1 = (1.0/ ( self.M + self.m - (self.m**2*self.l**2*ca.cos(x3)**2)/(self.I + self.m*self.l**2) ) ) * \
             (u + ( (self.m*self.l**2*ca.cos(x3)**2)/(self.I + self.m*self.l**2) - 1)*self.w  
              - self.bc*x2 
              - self.m*self.l*x4**2*ca.sin(x3) 
              + (self.m**2*self.l**2*self.g*ca.sin(x3)*ca.cos(x3))/(self.I + self.m*self.l**2) 
              - (self.m*self.l*self.bp*x4*ca.cos(x3))/(self.I + self.m*self.l**2) 
             )

        f2 = (1.0/ ( self.I + self.m*self.l**2 - (self.m**2*self.l**2*ca.cos(x3)**2)/(self.m + self.M) ) ) * \
             (
                 u*(self.m*self.l*ca.cos(x3))/(self.m+self.M) 
                 + self.w*(self.M*self.l*ca.cos(x3))/(self.m+self.M) 
                 - self.bp*x4 
                 + self.m*self.l*self.g*ca.sin(x3) 
                 - (self.m**2*self.l**2*x4**2*ca.sin(x3)*ca.cos(x3))/(self.m + self.M) 
                 - (self.m*self.l*self.bc*x2*ca.cos(x3))/(self.m+self.M)
             )
        
        dxdt = [ x2, f1, x4, f2 ]

        return ca.vertcat(*dxdt)

    def set_reference(self, ref):
        """
        Simple method to set the new system reference.

        :param ref: desired reference [m]
        :type ref: float or casadi.DM 1x1
        """
        self.x_d = ref
        
    def get_discrete_system_matrices_at_eq(self):
        """
        Evaluate the discretized matrices at the equilibrium point

        :return: A,B,C matrices for equilibrium point
        :rtype: casadi.DM 
        """
        A_eq = self.Ad(self.x_eq, self.u_eq)
        B_eq = self.Bd(self.x_eq, self.u_eq)
        
        # Populate a full observation matrix
        C_eq = ca.DM.zeros(1,4)
        C_eq[0,0] = 1
        C_eq[0,1] = 1
        C_eq[0,2] = 1
        C_eq[0,3] = 1

        return A_eq, B_eq, C_eq


    def continuous_time_linear_dynamics(self, x0, u):
        """
        Perform a time step iteration in continuous dynamics.

        :param x0: initial state
        :type x0: 4x1 ( list [a, b, c, d] , ca.MX )
        :param u: control input
        :type u: scalar, 1x1
        :return: dot(x), time derivative
        :rtype: 4x1, ca.DM
        """
        out = self.Integrator_lin(x0=x0, p=u)
        return out["xf"]

    def continuous_time_nonlinear_dynamics(self, x0, u):
        out = self.Integrator(x0=x0, p=u)
        return out["xf"]

    def discrete_time_dynamics(self,x0,u):
        """ 
        Performs a discrete time iteration step.

        :param x0: initial state
        :type x0: 4x1 ( list [a, b, c, d] , ca.MX )
        :param u: control input
        :type u: scalar, 1x1
        :return: next discrete time state
        :rtype: 4x1, ca.DM
        """

        return self.Ad(self.x_eq, self.u_eq) @ x0 + \
                self.Bd(self.x_eq, self.u_eq) @ u

    def enable_disturbance(self, w=0.01):
        """
        Enable system disturbance as a wind force.

        :param w: disturbance magnitude, defaults to 0.1
        :type w: float, optional
        """

        # Activate disturbance
        self.w = w

        # Re-generate integrators for dynamics with disturbance
        self.set_integrators()

    pass