import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.optimizer import Optimizer, required

# def f(x, y):
#     return (x**2 + y**2) + 5*np.sin(x + y**2)

# def df(x, y):
#     return (2*x + 5*np.cos(x + y**2), 2*y + 10*y*np.cos(x + y**2))  

def f(x, y):
    return (x**2 + y**2) + 5*np.sin(x + 2*y**2)

def df(x, y):
    return (2*x + 5*np.cos(x + 2*y**2), 2*y + 20*y*np.cos(x + 2*y**2))

class KO(Optimizer):
    def __init__(self, params, lr=required, weight_decay=0, sigma_Q=0.1, 
                 sigma_R=2, gamma=2, momentum=0.9, device=None):
        self.device = device
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, sigma_Q=sigma_Q, sigma_R=sigma_R, gamma=gamma, 
                        momentum=momentum, weight_decay=weight_decay)
        super(KO, self).__init__(params, defaults)

    def filter(self, y_t, x_t, P_t, sigma_Q, sigma_R, gamma):
        state_dim = x_t.shape[0]
        Q_t = sigma_Q * torch.eye(state_dim).to(self.device)
        R_t = sigma_R * torch.eye(state_dim).to(self.device)
        A_t = gamma * torch.eye(state_dim).to(self.device)
        C_t = torch.eye(state_dim).to(self.device)
        
        # Prediction Step
        x_t = torch.matmul(A_t, x_t)
        P_t = torch.matmul(A_t, torch.matmul(P_t, A_t)) + Q_t
        
        # Update Step
        r_t = y_t - torch.matmul(C_t, x_t)
        S_t = R_t + torch.matmul(C_t, torch.matmul(P_t, C_t))
        K_t = torch.matmul(P_t, torch.matmul(C_t, torch.inverse(S_t)))
        x_t = x_t + torch.matmul(K_t, r_t)
        P_t = torch.matmul((torch.eye(state_dim, device=self.device) - 
                            torch.matmul(K_t, C_t)), P_t)
        return x_t, P_t

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            sigma_R = group['sigma_R']
            sigma_Q = group['sigma_Q']
            gamma = group['gamma']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                param_state = self.state[p]
                if 'xt_buffer' not in param_state and 'Pt_buffer' not in param_state:
                    buf_x = param_state['xt_buffer'] = torch.clone(d_p).detach()
                    buf_P = param_state['Pt_buffer'] = torch.eye(d_p.shape[0], device=self.device) * 0.01
                else:
                    buf_x = param_state['xt_buffer']
                    buf_P = param_state['Pt_buffer']
                    buf_x, buf_P = self.filter(d_p, buf_x, buf_P, sigma_Q, sigma_R, gamma)
                    param_state['xt_buffer'] = buf_x
                    param_state['Pt_buffer'] = buf_P
                p.data.add_(-group['lr'], buf_x)
                
        return loss

# Initial point setup
x = torch.tensor([5.0], requires_grad=True)
y = torch.tensor([5.0], requires_grad=True)

num_iterations = 50
norm_1 = 0
norm_2 = 0
dis = 0
dis_2 = 0

opt_x = -1.11
opt_y = 0

x_dot = []
y_dot = []

optimizer_ko = KO([x, y], lr=0.1) #####################################change point

path_ko = [(x.item(), y.item())]

x_dot.append(x.item())
y_dot.append(y.item())  

# Optimization loop
for i in range(num_iterations):
    optimizer_ko.zero_grad()

    # Compute function value and gradient
    fx = f(x.item(), y.item())
    grad_fx = df(x.item(), y.item())

    # Convert gradients to torch tensors
    grad_x = torch.tensor(grad_fx[0])
    grad_y = torch.tensor(grad_fx[1])

    # Perform optimization steps
    optimizer_ko.step(closure=lambda: torch.tensor(fx).to(torch.float32).sum())

    # Update x and y based on gradients
    x = x - optimizer_ko.param_groups[0]['lr'] * grad_x
    y = y - optimizer_ko.param_groups[0]['lr'] * grad_y
    
    dis += ((x - opt_x) ** 2 + (y - opt_y) ** 2) ** (1/2)
    dis_2 += dis ** 2
    
    path_ko.append((x.item(), y.item()))
    
    if i % 5 == 0:
        x_dot.append(x.item())
        y_dot.append(y.item())                     

norm_1 = dis
norm_2 = dis ** (1/2)

print("1norm = ", norm_1)
print("2norm = ", norm_2)

x_dot.append(x.item())
y_dot.append(y.item())

# Convert paths to numpy arrays for plotting
path_ko = np.array(path_ko)

# Extract x and y coordinates for contour plot
x = np.arange(-10, 10, 0.01)
y = np.arange(-10, 10, 0.01)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

mask = Z > 25 
Z[mask] = 0

# Contour plot
plt.contour(X, Y, Z)
plt.plot(path_ko[:, 0], path_ko[:, 1], color='red', linewidth=2, label='KO Path')
plt.plot(x_dot, y_dot, 'o', color = "black", markersize = 4)
plt.ylim(-5, 5)
plt.xlim(-5, 5)
plt.plot(-1.11, 0, 'x')
plt.title('KO : f(x,y) = x**2 + y**2 + 5sin(x + 2*y**2)\nlr = 0.1      init_point(5, 5)')
plt.xlabel('x')
plt.ylabel('y')
print(f"{x_dot[-1]:.4f}, {y_dot[-1]:.4f}")
plt.show()
