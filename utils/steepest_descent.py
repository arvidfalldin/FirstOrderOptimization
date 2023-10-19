"""
Collection of implementations of various (deterministic) steepest descent
algorithms. 
"""


def steepest_descent(f,
                     grad_f,
                     x0,
                     alpha,
                     epsilon=1e-6,
                     max_iter=1000,
                     ):
    """
    Naive implementation of the "vanilla" steepest descent method using a fixed
    step length alpha.
    INPUT:
    f           -   objective function (callable)
    grad_f      -   gradient of f (callable)
    x0          -   inital guess
    alpha       -   (fixed) step length
    max_iter    -   maximum number if iterations before break

    OUPUT:
    x_iter      -   list of iterations of x^k
    f_iter      -   list of function values f_iter[i] = f(x_iter[i])

    """
    # TODO: Add optional stopping criteria. E.g norm of grad

    # Init lists of x- anf f-iterates
    x_iter = [x0,]
    f_iter = [f(x0),]

    # Steepest descent update scheme
    for i in range(max_iter):
        if f_iter[i] < epsilon:
            break
        # Apply steepest descent update
        x_iter.append(x_iter[i] - alpha*grad_f(x_iter[i]))

        # Eval objective function at new point
        f_iter.append(f(x_iter[i+1]))

    if i == max_iter-1:
        # TODO: Raise a proper warning if we reach maxiter without convergence
        print("Warning! Did not converge")
        pass
    return x_iter, f_iter


def steepest_descent_with_line_search(f,
                                      grad_f,
                                      x0,
                                      alpha,
                                      epsilon,
                                      max_iter):
    """
    Steepest descent with exact line search.

    INPUT:
    f           -   objective function (callable)
    grad_f      -   gradient of f (callable)
    x0          -   inital guess
    alpha       -   dynamical step length as function of x (callable)
    max_iter    -   maximum number if iterations before break

    OUPUT:
    x_iter      -   list of iterations of x^k
    f_iter      -   list of function values f_iter[i] = f(x_iter[i])
    """

    # Init lists of x- anf f-iterates
    x_iter = [x0,]
    f_iter = [f(x0),]

    # Steepest descent update scheme
    for i in range(max_iter):
        if f_iter[i] < epsilon:
            break

        # Apply steepest descent update with exact line search
        x_iter.append(x_iter[i] - alpha(x_iter[i])*grad_f(x_iter[i]))

        # Eval objective function at new point
        f_iter.append(f(x_iter[i+1]))

    if i == max_iter-1:
        # TODO: Raise a proper warning if we reach maxiter without convergence
        print("Warning! Did not converge")
        pass
    return x_iter, f_iter


def heavy_ball(f,
               grad_f,
               x0,
               alpha,
               beta,
               epsilon=1e-6,
               max_iter=1000,
               ):
    """
    Simple implementation of the Heavy ball steepest descent method

    INPUT:
    f           -   objective function (callable)
    grad_f      -   gradient of f (callable)
    x0          -   inital guess
    alpha, beta -   heavy ball parameters
    max_iter    -   maximum number if iterations before break

    OUPUT:
    x_iter      -   list of iterations of x^k
    f_iter      -   list of function values f_iter[i] = f(x_iter[i])
    """

    # Init lists of iterates
    x_iter = [x0, x0]  # Heavy ball looks two steps back in time
    f_iter = [f(x0),]

    # Steepest descent update scheme
    for i in range(1, max_iter+1):
        if f_iter[i-1] < epsilon:
            break

        # Apply heavy ball steepest descent update
        x_iter.append(x_iter[i]
                      - alpha*grad_f(x_iter[i])
                      + beta*(x_iter[i] - x_iter[i-1]))

        # Eval objective function at new point
        f_iter.append(f(x_iter[i+1]))

    # Remove the duplicate entry in the beginning of the list
    x_iter.pop(0)

    if i == max_iter-1:
        # TODO: Raise a proper warning if we reach maxiter without convergence
        print("Warning! Did not converge")
        pass
    return x_iter, f_iter


def nestorovs_optimal(f,
                      grad_f,
                      x0,
                      alpha,
                      beta,
                      epsilon=1e-6,
                      max_iter=1000,
                      ):
    """
    Simple implementation of the Heavy ball steepest descent method

    INPUT:
    f           -   objective function (callable)
    grad_f      -   gradient of f (callable)
    x0          -   inital guess
    alpha, beta -   heavy ball parameters
    max_iter    -   maximum number if iterations before break

    OUPUT:
    x_iter      -   list of iterations of x^k
    f_iter      -   list of function values f_iter[i] = f(x_iter[i])
    """

    # Init lists of iterates
    x_iter = [x0, x0]  # Nestorov looks two steps back in time
    f_iter = [f(x0),]

    # Steepest descent update scheme
    for i in range(1, max_iter + 1):
        if f_iter[i-1] < epsilon:
            break

        # Apply heavy ball update
        delta = x_iter[i] - x_iter[i-1]
        x_iter.append(x_iter[i]
                      - alpha*grad_f(x_iter[i] + beta*delta)
                      + beta*delta)

        # Eval objective function at new point
        f_iter.append(f(x_iter[i+1]))

    # Remove the duplicate entry in the beginning of the list
    x_iter.pop(0)

    if i == max_iter-1:
        # TODO: Raise a proper warning if we reach maxiter without convergence
        print("Warning! Did not converge")
        pass
    return x_iter, f_iter
