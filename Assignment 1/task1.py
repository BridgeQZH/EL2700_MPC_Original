from model import Pendulum
from controller import Controller
from simulation import EmbeddedSimEnvironment

# Create pendulum and controller objects
pendulum = Pendulum()
ctl = Controller()

# Get the system discrete-time dynamics
A, B, C = pendulum.get_discrete_system_matrices_at_eq()

# Get control gains
ctl.set_system(A, B, C)
K = ctl.get_closed_loop_gain()
lr = ctl.get_feedforward_gain(K)

# Initialize simulation environment
sim_env = EmbeddedSimEnvironment(model=pendulum, 
                                dynamics=pendulum.discrete_time_dynamics,
                                controller=ctl.control_law,
                                time = 20)

# Enable model disturbance for second simulation environment
pendulum.enable_disturbance(w=0.01)
sim_env_with_disturbance = EmbeddedSimEnvironment(model=pendulum, 
                                dynamics=pendulum.continuous_time_nonlinear_dynamics,
                                controller=ctl.control_law,
                                time = 20)

# Also returns time and state evolution
t, y, u = sim_env.run([0,0,0,0])
t, y, u = sim_env_with_disturbance.run([0,0,0,0])