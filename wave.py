# http://www.aosocean.com/en/article/doi/10.1007/s13131-021-1871-6
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

g = 9.81
omega0 = 1
theta0 = 1

dtheta = 1/100
domega = 1/10

Theta = np.arange(-0.5*np.pi, 0.5*np.pi, dtheta)
Omega = np.arange(0.01, 6, domega)

a=0.01
U=5
beta=1.25

def jonswap(omega, theta):
    p = 0.5+0.82*np.exp(-0.5*(omega/8.565*U)*4)
    q = (p-0.5)/0.82*0.32
    return a*g**2/omega**5 * np.exp(-beta*(g/U/omega)**4) * 1/np.pi * (1+p*np.cos(2*theta)+q*np.cos(4*theta))

def wavePlot(X, Y):
    assert(X.shape == Y.shape)
    # Z = rng.random(size=X.shape)
    # Z = np.zeros_like(X)
    Z = 0.1*np.sin(X/3)
    t = 245

    Xout = X
    Yout = Y
    Zout = Z

    for oi in Omega:
        for tj in Theta:
            r = np.sqrt(jonswap(oi, tj)*dtheta*domega)
            phase = oi**2/g * (X*np.cos(tj)+Y*np.sin(tj) - oi*t)
            Xout = Xout - r*np.sin(phase)
            Yout = Yout - r*np.sin(phase)
            Zout = Zout + r*np.cos(phase)
    return Xout, Yout, Zout

x = np.arange(0.0, 100.0, 1)
X, Y = np.meshgrid(x, x)
# X, Y = np.meshgrid(Theta,Omega)

X, Y, Z = wavePlot(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z)
# surf = ax.plot_surface(X, Y, jonswap(Y, X))
fig.colorbar(surf)
ax.set_title("Surface Plot")
ax.set_zlim(-2,2)
plt.show()

