import numpy as np
import time


def delay(x, t, theta):
    """
    Shifts a vector x with theta in time
    """
    return np.interp(t - theta, t, x)


def residual(a, b, theta, u, y, t):
    """
    Calculates the residuals for a given set of parameters and data.
    """
    y_k0 = y[:-1]
    y_k1 = y[1:]

    dt = (t[1:] - t[:-1])

    # Calculate residuals
    res = -y_k1 + (1 + dt*a) * y_k0 + dt*b * delay(u, t, theta)[:-1]

    # add dimension the array because numpy
    return np.array([res]).T


def J_residual(a, b, theta, u, y, t):
    """
    Calculates the jacobian of the residuals
    """
    dt = (t[1:] - t[:-1])
    y_k0 = y[:-1]
    y_k1 = y[1:]

    # Shift input and derivative by theta and calculate Jacobian matrix
    u_shift = delay(u, t, theta)[:-1]
    du_shift = delay(np.gradient(u, t), t, theta)[:-1]

    J_res = np.array([dt*y_k0, dt*u_shift, dt*b*du_shift*(-1)]).T

    return J_res


def objective(a, b, theta, u, y, t):
    """
    Calculates the objective function for a given set of parameters and data.
    """
    r = residual(a, b, theta, u, y, t)
    return 0.5*np.sum(r**2)


def J_objective(a, b, theta, u, y, t):
    """
    Calculates the Jacobian of the objective
    """
    res = residual(a, b, theta, u, y, t)
    J_res = J_residual(a, b, theta, u, y, t)

    return res.T @ J_res


def step(a, b, theta, delta, alpha):
    """
    Changes updates the decision variables based on the direction and size
    """
    a_0 = a + alpha * delta[0]
    b_0 = b + alpha * delta[1]
    theta_0 = theta + alpha * delta[2]

    return a_0[0], b_0[0], theta_0[0]


def armijo_cond(S_0, S_1, J_s, delta, alpha, c=0.5):
    """
    Checks if Armiho condition is met
    """
    return S_1 < (S_0 + c * alpha * J_s @ delta)


def line_search(a, b, theta, u, y, t, delta, c):
    """
    Performs a line search to find the step size alpha
    """
    S0 = objective(a, b, theta, u, y, t)
    J_s = J_objective(a, b, theta, u, y, t)

    alpha = 1
    while (alpha > 1e-10):
        S1 = objective(*step(a, b, theta, delta, alpha), u, y, t)
        # check if armijo condition is met
        if armijo_cond(S0, S1, J_s, delta, alpha, c):
            return alpha
        else:
            # reduce alpha if condition is not met
            alpha /= 2

    return 0


def convert_to_a_b(tua, K):
    """
    Converts the parameters to the form used in the model
    """
    a = -1/tua
    b = K/tua
    return a, b


def convert_to_tua_K(a, b):
    """
    Converts the parameters to the form used in the optimization problem
    """
    tua = -1/a
    K = b*tua
    return tua, K


def fit_model(tua_0, K_0, theta_0, u, y, t, max_iters=150, c=0.5, verbose=False):
    """
    Solves an optimization problem to find the values of the parameters `tua`, `K`, and `theta`
    that minimize the sum of squared residuals between the input `u`, its time derivative `u_dot`,
    and the output `y`.
    """
    start_time = time.time()

    # Define initial parameter values
    # tua = tua_0
    # K = K_0
    a, b = convert_to_a_b(tua_0, K_0)
    theta = theta_0

    iter_count = 0
    end_reached = False
    alpha = -1

    # Loop until maximum number of iterations is reached or line search fails
    while iter_count < max_iters and not end_reached:
        # Calculate current objective function and parameter values
        S = objective(a, b, theta, u, y, t)

        if verbose:
            # Print current parameter values and objective function
            tua, K = convert_to_tua_K(a, b)
            print(
                f"Iteration {iter_count}: S = {S:.5f}, tua = {tua:.5f}, K = {K:.5f}, theta = {theta:.5f}, alpha = {alpha:.5f}")

        # Calculate the Jacobian of the residuals
        J_res = J_residual(a, b, theta, u, y, t)

        # Calculate the residuals
        r = residual(a, b, theta, u, y, t)

        # Calculate the Newton step
        # add regularization to avoid singular matrix
        delta = np.linalg.solve(J_res.T @ J_res + 1e-5*np.eye(3), -J_res.T @ r)

        # apply line search for globalization
        alpha = line_search(a, b, theta, u, y, t, delta, c=c)

        # Check if line search failed
        if alpha == 0:
            end_reached = True
            break

        # Update parameter values and iteration count
        a, b, theta = step(a, b, theta, delta, alpha)

        iter_count += 1

    # Print solution and total time taken to solve optimization problem
    if verbose:
        tua, K = convert_to_tua_K(a, b)
        print(f"\nSolution: tua = {tua:.5f}, K = {K:.5f}, theta = {theta:.5f}")
        print(f"Total time taken: {time.time() - start_time:.5f} seconds")

    return tua, K, theta
