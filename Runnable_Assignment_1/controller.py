import casadi as ca
import numpy as np
from control.matlab import place

class Controller(object):

    def __init__(self):
        """
        Controller class. Implements the controller:
        u_t = - L @ x_t + lr @ r
        """

        # Do not consider the disturbance
        # self.p1 = 0.95
        # self.p2 = 0.94
        # self.p3 = 0.91
        # self.p4 = 0.90

        # Consider the disturbance input
        self.p1 = 0.94
        self.p2 = 0.91
        self.p3 = 0.85+0.1j
        self.p4 = 0.85-0.1j

        self.poles = [self.p1, self.p2, self.p3, self.p4]
        self.L = ca.DM.zeros(1,4)

        # print(self)                             # You can comment this line

    def __str__(self):
        return """
            Controller class. Implements the controller:
            u_t = - L @ x_t + lr @ r
        """

    def set_system(self, A=ca.DM.zeros(4,4), B=ca.DM.zeros(4,1), \
                         C=ca.DM.zeros(1,4), D=ca.DM.zeros(1,1)):
        """
        Set system matrices.

        :param A: state space A matrix
        :type A: casadi.DM
        :param B: state space B matrix
        :type B: casadi.DM
        """
        self.A = A
        self.B = B
        self.C = C
        self.D = D


    def get_closed_loop_gain(self, p=None):
        """
        Get the closed loop gain for the specified poles.

        :param p: pole list, defaults to self.p
        :type p: [type], optional
        :return: [description]
        :rtype: [type]
        """
        if p is None:
            p = self.poles

        A = np.array(self.A).tolist()
        B = np.array(self.B).tolist()

        # L = place(self.A,self.B,self.poles)
        L = place(A,B,p)
        # Set L as ca.DM
        self.L = ca.DM.zeros(1,4)
        self.L[0,0] = L[0,0] 
        self.L[0,1] = L[0,1]
        self.L[0,2] = L[0,2]
        self.L[0,3] = L[0,3]
        
        return self.L

    def set_poles(self, p, p2=None, p3=None, p4=None):
        """
        Set closed loop poles. If 'p' is a list of poles, then the remaining
        inputs are ignored. Otherwise, [p,p2,p3,p4] are set as poles.

        :param p: pole 1 or pole list
        :type p: list or scalar
        :param p2: pole 2, defaults to None
        :type p2: scalar, optional
        :param p3: pole 3, defaults to None
        :type p3: scalar, optional
        :param p4: pole 4, defaults to None
        :type p4: scalar, optional
        """

        if isinstance(p, list):
            self.poles = p
        else:
            self.p1 = p
            self.p2 = p2
            self.p3 = p3
            self.p4 = p4
            self.poles = [self.p1, self.p2, self.p3, self.p4]

    def get_feedforward_gain(self, L=None):
        """
        Get the feedforward gain lr.

        :param L: close loop gain, defaults to None
        :type L: list, optional
        """

        if L is None:
            L = self.L

        self.lrd = self.C @ ca.inv(ca.DM.eye(4) - (self.A - self.B @ self.L)) @ self.B
        self.lr = 1.0 / self.lrd
        return self.lr

    def control_law(self, x, ref=10.0):
        """
        Nonlinear control law.

        :param x: state
        :type x: casadi.DM
        :param ref: cart reference position, defaults to 10
        :type ref: float, optional
        """
        u = -self.L @ x + self.lr @ ref
        
        return u