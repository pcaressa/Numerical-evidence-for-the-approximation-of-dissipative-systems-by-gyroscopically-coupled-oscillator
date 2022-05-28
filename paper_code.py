import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import dual_annealing, OptimizeResult

def simulate_damped(tt, t, y_damped, omega, gamma, x0, N_MAX = 11):
    """According to tt ("Kdiagonal", "Kscalar", "Kconstant")
    performs the optimization against the y_damped curve, in the t
    time interval, being omega, gamma and x0 the parameters and
    initial displacement of the damped motion (initial velocity is
    zero, masses are 1). Performs the simulations 3,5,...,N_MAX
    degrees of freedom: the result is a list of triples [N0,
    parameters , y] where parameters are the optimized parameters
    and y is the resulting approximating curve."""

    yy = []
    previous_loss = 1e20
    for N in range(3,N_MAX):
        print()
        N2 = N*2

        # Choose the functions/bounds according to the tt parameter
        # since we use N and NN inside the gyroscopic() functions
        # we need to define them here because of Python lexical
        # scoping rules.
        if tt == "Kdiagonal":
            def gyroscopic(y, t, *p):
                """Returns the N2 second members of the 1st order
                ODE system which implements the gyroscopic coupling
                where p[0:N-1] are the coefficient of the
                superdiagonal of G and p[N-1:2N-2] are the
                diagonal of K."""
                e = np.zeros((N2,))
                # NB: y[0:N] are q and y[N:] are p
                e[0] = y[N] - p[0]*y[1]
                for k in range(1,N-1):
                    e[k] = y[N+k] + p[k-1]*y[k-1] - p[k]*y[k+1]
                e[N-1] = y[N2-1] + p[N-2]*y[N-2]
                e[N] = -(0.5*p[0]**2 + p[N-1])*y[0] + 0.5*p[0]*p[1]*y[2] - p[0]*y[N+1]
                e[N+1] = -(0.5*p[0]**2 + 0.5*p[1]**2 + p[N])*y[1] + 0.5*p[1]*p[2]*y[3] + p[0]*y[N-1] - p[1]*y[N+1]
                for h in range(2, N-2):
                    e[N+h] = -(0.5*p[h-1]**2 + 0.5*p[h]**2 + p[N-1+h])*y[h] + 0.5*p[h-2]*p[h-1]*y[h-2] + 0.5*p[h]*p[h+1]*y[h+2]+p[h-1]*y[N+h-2] -p[h]*y[N+h]
                e[N2-2] = -(0.5*p[N-3]**2 + 0.5*p[N-2]**2 + p[N2-3])*y[N-2] + 0.5*p[N-4]*p[N-3]*y[N-4]
                e[N2-1] = -(0.5*p[N-2]**2 + p[N2-2])*y[N-1] + 0.5*p[N-3]*p[N-2]*y[N-3]
                return e
            bounds = ([(gamma/10,gamma*10)] *(N-1)) + ([(omega2/10,omega2*10)] *N)
        elif tt == "Kscalar":
            if N == 3:
                def gyroscopic(y, t, *p):
                    """Returns the N2 second members of the 1st
                    order ODE system which implements the
                    gyroscopic coupling, where p[0:N-1] are
                    the coefficient of the superdiagonal of G and
                    p[N-1:2N-2] are the diagonal of K."""
                    e = np.zeros((6,))
                    # NB: y[0:N] are q and y[N:] are p
                    e[0] = y[3] - p[0]*y[1]
                    e[1] = y[4] + p[0]*y[0] - p[1]*y[2]
                    e[2] = y[5] + p[1]*y[1]
                    e[3] = -(0.5*p[0]**2 + p[2])*y[0] + 0.5*p[0]*p[1]*y[2] - p[0]*y[4]
                    e[4] = -(0.5*p[0]**2 + 0.5*p[1]**2 + p[2])*y[1]
                    e[5] = -(0.5*p[1]**2 + p[2])*y[2] + 0.5*p[0]*p[1]*y[0]
                    return e
            else:
                def gyroscopic(y, t, *p):
                    """Returns the N2 second members of the 1st
                    order ODE system which implements the
                    gyroscopic coupling, where p[0:N-1] are the
                    coefficient of the superdiagonal of G and
                    p[N-1:2N-2] are the diagonal of K."""
                    e = np.zeros((N2,))
                    # NB: y[0:N] are q and y[N:] are p
                    e[0] = y[N] - p[0]*y[1]
                    for k in range(1,N-1):
                        e[k] = y[N+k] + p[k-1]*y[k-1] - p[k]*y[k+1]
                    e[N-1] = y[N2-1] + p[N-2]*y[N-2]
                    e[N] = -(0.5*p[0]**2 + p[N-1])*y[0] + 0.5*p[-1]*p[1]*y[2] - p[0]*y[N+1]
                    e[N+1] = -(0.5*p[0]**2 + 0.5*p[1]**2 + p[-1])*y[1] + 0.5*p[1]*p[2]*y[3] + p[0]*y[N-1] - p[1]*y[N+1]
                    for h in range(2, N-2):
                        e[N+h] = -(0.5*p[h-1]**2 + 0.5*p[h]**2 + p[-1])*y[h] + 0.5*p[h-2]*p[h-1]*y[h-2] + 0.5*p[h]*p[h+1]*y[h+2]+p[h-1]*y[N+h-2] -p[h]*y[N+h]
                    e[N2-2] = -(0.5*p[N-3]**2 + 0.5*p[N-2]**2 + p[-1])*y[N-2] + 0.5*p[N-4]*p[N-3]*y[N-4]
                    e[N2-1] = -(0.5*p[N-2]**2 + p[-1])*y[N-1] + 0.5*p[N-3]*p[N-2]*y[N-3]
                    return e
            bounds = [(gamma/10,gamma*10)] *(N-1) + [(omega2/10,omega2*10)]
        elif tt == "Kconstant":
            if N == 3:
                def gyroscopic(y, t, *p):
                    """Returns the N2 second members of the 1st
                    order ODE system which implements the
                    gyroscopic coupling, where p[0:N-1] are the
                    coefficient of the superdiagonal of G and
                    p[N-1:2N-2] are the diagonal of K."""
                    e = np.zeros((6,))
                    # NB: y[0:N] are q and y[N:] are p
                    e[0] = y[3] - p[0]*y[1]
                    e[1] = y[4] + p[0]*y[0] - p[1]*y[2]
                    e[2] = y[5] + p[1]*y[1]
                    e[3] = -(0.5*p[0]**2 + omega2)*y[0] + 0.5*p[0]*p[1]*y[2] - p[0]*y[4]
                    e[4] = -(0.5*p[0]**2 + 0.5*p[1]**2 + omega2)*y[1]
                    e[5] = -(0.5*p[1]**2 + omega2)*y[2] + 0.5*p[0]*p[1]*y[0]
                    return e
            else:
                def gyroscopic(y, t, *p):
                    """Returns the N2 second members of the 1st
                    order ODE system which implements the
                    gyroscopic coupling, where p[0:N-1] are the
                    coefficient of the superdiagonal of G and
                    p[N-1:2N-2] are the diagonal of K."""
                    e = np.zeros((N2,))
                    # NB: y[0:N] are q and y[N:] are p
                    e[0] = y[N] - p[0]*y[1]
                    for k in range(1,N-1):
                        e[k] = y[N+k] + p[k-1]*y[k-1] - p[k]*y[k+1]
                    e[N-1] = y[N2-1] + p[N-2]*y[N-2]
                    e[N] = -(0.5*p[0]**2 + omega2)*y[0] + 0.5*p[0]*p[1]*y[2] - p[0]*y[N+1]
                    e[N+1] = -(0.5*p[0]**2 + 0.5*p[1]**2 + omega2)*y[1] + 0.5*p[1]*p[2]*y[3] + p[0]*y[N-1] - p[1]*y[N+1]
                    for h in range(2, N-2):
                        e[N+h] = -(0.5*p[h-1]**2 + 0.5*p[h]**2 + omega2)*y[h] + 0.5*p[h-2]*p[h-1]*y[h-2] + 0.5*p[h]*p[h+1]*y[h+2]+p[h-1]*y[N+h-2] -p[h]*y[N+h]
                    e[N2-2] = -(0.5*p[N-3]**2 + 0.5*p[N-2]**2 + omega2)*y[N-2] + 0.5*p[N-4]*p[N-3]*y[N-4]
                    e[N2-1] = -(0.5*p[N-2]**2 + omega2)*y[N-1] + 0.5*p[N-3]*p[N-2]*y[N-3]
                    return e
            bounds = [(gamma/10,gamma*10)] *(N-1)
        else:
            raise "Bad tt parameter"

        print(N, "degrees of freedom optimization with bounds", bounds)

        # Initial conditions to solve the system
        init_cond = np.zeros((N2,))
        init_cond[0] = x0

        def loss(params):
            y = odeint(gyroscopic, init_cond, t, args = tuple(params))[:,0]
            return np.linalg.norm(y - y_damped, ord=np.inf)  # sup norm

        def annealing_callback(x, f, context):
            if context == 0:
                print("... Minimum found! Loss:", loss(x))
            return False
        m_best = OptimizeResult()
        l_best = 1e20
        for i in range(N):
            print(f"... Optimization [{i+1}/{N}]-th trial")
            m = dual_annealing(loss, bounds, callback = annealing_callback)
            # Compare with previous loss: repeat if no improvement
            l = loss(m.x)
            if l < l_best:
                l_best = l
                m_best = m
            if l < previous_loss:
                previous_loss = l
                break

        optima = m_best.x
        y = odeint(gyroscopic, init_cond, t, args = tuple(optima))[:,0]
        yy.append((N, optima, y))
        print()
        print("Minimization", "ok." if m.status else "KO!", m.message)
        print("Loss: ", loss(optima))
        print("Result [omega_k**2, omega_{hk}]:", optima)
    return yy

def result_plot(t, y_damped, yy, filename = None):
    plt.figure(figsize=(16, 9))
    plt.rcParams["font.size"] = "16"
    plt.grid()
    for record in yy:
        plt.plot(t,record[2], label=f"$N={record[0]}$")
    plt.plot(t,y_damped, "black", label="damped")
    plt.legend()
    plt.savefig(filename + ".png")
    plt.show()

    plt.figure(figsize=(16, 9))
    plt.rcParams["font.size"] = "24"
    ascisse = range(yy[0][0],yy[-1][0]+1)
    errori = [np.linalg.norm(y_damped - np.array(yy[i][2]), ord=np.inf) for i in range(len(yy))]

    print(ascisse)
    print(errori)

    plt.xticks(ascisse)
    plt.yticks(np.linspace(0,errori[0],len(errori)))
    plt.grid()
    plt.plot(ascisse,errori,"-o")
    plt.savefig(filename + "_error.png")
    plt.show()

    # Choose the minimum error and shows its parameters
    imin = 0
    errore = 1e20
    for i in range(len(errori)-1, -1, -1):
        if errori[i] < errore:
            errore = errori[i]
            imin = i
    N = yy[imin]
    print("Errors:", errori)
    print(f"Best approximation for N = {N} (loss = {errore:.4})")
    print(f"omega = {np.sqrt(yy[imin][1][0]):.4}")

    plt.figure(figsize=(16, 9))
    plt.rcParams["font.size"] = "24"
    plt.grid()
    plt.plot(t,yy[imin][2], label=f"$N={yy[imin][0]}$")
    plt.plot(t,y_damped, "black", label="damped")
    plt.legend()
    if filename != None:
        plt.savefig(filename + ".png")
    else:
        plt.show()

t = np.linspace(0, 5, 100)  # Time interval sample considered

# Parameters of the damped system
gamma = 0.5
omega = 4
x0 = 1
p0 = 0
a = 2*gamma
omega2 = omega*omega

# Computes the solution y_damped of x''+gamma x' + omega^2 x = 0
y_damped = odeint(lambda y, t: [y[1], -a*y[1] - omega2*y[0]], [x0, p0], t)[:,0]

# Computes the solution y_damped2 of x''+gamma x'|x'| + omega^2 sin x = 0
y_damped2 = odeint(lambda y, t: [y[1], -a*y[1]*abs(y[1]) - omega2*np.sin(y[0])], [x0, p0], t)[:,0]

print("Gyroscopic coupling, K diagonal, simple damping.")
yy = simulate_damped("Kdiagonal", t, y_damped, omega, gamma, x0)
result_plot(t, y_damped, yy, "gyroscopic_Kdiagonal")
print(yy)

print("Gyroscopic coupling, K scalar, simple damping.")
yy = simulate_damped("Kscalar", t, y_damped, omega, gamma, x0)
result_plot(t, y_damped, yy, "gyroscopic_Kscalar")
print(yy)

print("Gyroscopic coupling, K constant, simple damping.")
yy = simulate_damped("Kconstant", t, y_damped, omega, gamma, x0)
result_plot(t, y_damped, yy, "gyroscopic_Kconstant")
print(yy)
