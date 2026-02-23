import numpy as np
from jaxtyping import Float

def undistort(point: Float[np.ndarray, "2 1"], distortion_center: Float[np.ndarray, "2 1"], k1: float, k2: float, k3: float, iterations = 4000) -> Float[np.ndarray, "2 1"]:
    """
    Desperate attempt at getting the correct 2D-Pixel coordinate point via undistortion using the given params.

    Allows various methods to be tried out.
    """
    #These are likely incorrect:
    #return undistort_iter_over(point, distortion_center, k1, k2, k3, iterations)
    #return undistort_newton_disabled(point, distortion_center, k1, k2, k3, iterations)

    # So far these two work the best, producing identical results.
    return undistort_one_shot_no_sqrt(point, distortion_center, k1, k2, k3, iterations)
    #return undistort_one_shot(point, distortion_center, k1, k2, k3, iterations)

def undistort_newton_disabled(point: Float[np.ndarray, "2 1"], distortion_center: Float[np.ndarray, "2 1"], k1: float, k2: float, k3: float, iterations = 4000) -> Float[np.ndarray, "2 1"]:
    """
    Performs distortion correction according to the "Division Model" by solving the equation for x_u and y_u numerically. 

    Uses Newton's method. 
    """

    x_u = point[[0]]
    y_u = point[1]
    
    x_c = distortion_center[0]
    y_c = distortion_center[1]

    x_d = point[0]
    y_d = point[1]

    #r = math.sqrt((x_u - x_c)**2 + (y_u - y_c)**2)
    #f = lambda x_u, x_d, x_c: x_c + (x_d - x_c)/(1 + k1 * ((x_u - x_c)**2 + (y_u - y_c)**2) + k2 * ((x_u - x_c)**2 + (y_u - y_c)**2)**2 + k3 * ((x_u - x_c)**2 + (y_u - y_c)**2)**3) - x_u
    #f_prime = lambda x_u, x_d, x_c: - (x_d - x_c) * ((x_u - x_c) * (2 * k1 + 4 * k2 * r ** 2 + 6 * k3 * r**4))/((1 + k1 * r**2 + k2 * r**4 + k3 * r**6))**2 -1

    f = lambda x_u, x_d, x_c, r: x_c + (x_d - x_c)/(1 + k1 * r**2 + k2 * r**4 + k3 * r**6) - x_u #  "Division" Formula for undistortion -> x_u, reordered to be equivalent to 0
    f_prime = lambda x_u, x_d, x_c, r: - (x_d - x_c) * ((x_u - x_c) * (2 * k1 + 4 * k2 * r ** 2 + 6 * k3 * r**4))/((1 + k1 * r**2 + k2 * r**4 + k3 * r**6))**2 -1
    
    r_hist = []
    
    for _ in range(iterations):
        r = np.sqrt((x_u - x_c)**2 + (y_u - y_c)**2)
        x_u = x_u - f(x_u, x_d, x_c, r) / f_prime(x_u, x_d, x_c, r)
        y_u = y_u - f(y_u, y_d, y_c, r) / f_prime(y_u, y_d, y_c, r)

        r_hist.append(np.sqrt((x_u - x_c)**2 + (y_u - y_c)**2))
    return np.reshape(np.array([x_u, y_u]), (2,1))

def undistort_newton_inverse_disabled(point: Float[np.ndarray, "2 1"], distortion_center: Float[np.ndarray, "2 1"], k1: float, k2: float, k3: float, iterations = 4000) -> Float[np.ndarray, "2 1"]:
    """
    Performs distortion correction according to the inverse Brown Conrady Model by solving the equation for x_u and y_u numerically. 

    Uses Newton's method. 
    """

    x_u = point[0]
    y_u = point[1]
    
    x_c = distortion_center[0]
    y_c = distortion_center[1]

    x_d = point[0]
    y_d = point[1]

    #r = math.sqrt((x_u - x_c)**2 + (y_u - y_c)**2)
    #f = lambda x_u, x_d, x_c: x_c + (x_d - x_c)/(1 + k1 * ((x_u - x_c)**2 + (y_u - y_c)**2) + k2 * ((x_u - x_c)**2 + (y_u - y_c)**2)**2 + k3 * ((x_u - x_c)**2 + (y_u - y_c)**2)**3) - x_u
    #f_prime = lambda x_u, x_d, x_c: - (x_d - x_c) * ((x_u - x_c) * (2 * k1 + 4 * k2 * r ** 2 + 6 * k3 * r**4))/((1 + k1 * r**2 + k2 * r**4 + k3 * r**6))**2 -1

    f = lambda x_u, x_d, x_c, r: x_d + (x_d - x_c) * (k1 * (r ** 2) + k2 * (r ** 4) + k3 * (r ** 6)) - x_u #  Brown Conrady Formula for undistortion, reordert to be equivalent to 0
    f_prime = lambda x_u, x_d, x_c, r: (x_u - x_c) * (x_d - x_c) * (2 * k1 + 4 * k2 * r ** 2 + 6 * k3 * r ** 4) - 1
    
    r_hist = []
    
    for _ in range(iterations):
        r = np.sqrt((x_u - x_c)**2 + (y_u - y_c)**2)
        x_u_prev = x_u
        y_u_prev = y_u

        x_u = x_u_prev - f(x_u_prev, x_d, x_c, r) / f_prime(x_u_prev, x_d, x_c, r)
        y_u = y_u_prev - f(y_u_prev, y_d, y_c, r) / f_prime(y_u_prev, y_d, y_c, r)

        r_hist.append(np.sqrt((x_u - x_c)**2 + (y_u - y_c)**2))
    return np.reshape(np.array([x_u, y_u]), (2,1))

def undistort_newton_disabled_ii(point: Float[np.ndarray, "2 1"], distortion_center: Float[np.ndarray, "2 1"], k1: float, k2: float, k3: float, iterations = 4000) -> Float[np.ndarray, "2 1"]:
    """
    Performs distortion correction according to the forward Brown Conrady Model by solving the equation for x_u and y_u numerically. 

    Uses Newton's method solving f(x_u, y_u) = (x_d , y_d)
    """

    x_u = point[0]
    y_u = point[1]
    
    x_c = distortion_center[0]
    y_c = distortion_center[1]

    x_d = point[0]
    y_d = point[1]

    r = np.sqrt((x_u - x_c)**2 + (y_u - y_c)**2)
    f = lambda x_u, x_d, r: x_u + x_u * (k1 * (r ** 2) + k2 * (r ** 4) + k3 * (r ** 6)) - x_d# forward distortion to equal zero
    f_prime = lambda x_u, r: (
        1
        + k1 * r**2
        + k2 * r**4
        + k3 * r**6
        + x_u * (x_u - x_c) * (2 * k1 + 4 * k2 * r**2 + 6 * k3 * r**4)
    )

    
    r_hist = []
    
    for _ in range(iterations):
        r = np.sqrt((x_u - x_c)**2 + (y_u - y_c)**2)
        x_u = x_u - f(x_u, x_d, r) / f_prime(x_u, r)
        y_u = y_u - f(y_u, y_d, r) / f_prime(y_u, r)

        r_hist.append(np.sqrt((x_u - x_c)**2 + (y_u - y_c)**2))
    return np.reshape(np.array([x_u, y_u]), (2,1))

def undistort_newton_chatty(
    point: np.ndarray,               # shape (2, 1) or (2,)
    distortion_center: np.ndarray,    # shape (2, 1) or (2,)
    k1: float,
    k2: float,
    k3: float,
    max_iter: int = 10,
    tol: float = 1e-12
) -> np.ndarray:
    """
    Undistort a point using Newton's method and the forward Brown–Conrady model
    (radial distortion only).

    implementation by yours truly: chatgpt
    Returns the undistorted point (x_u, y_u).
    """

    # --- unpack ---
    x_d, y_d = point[0], point[1]
    x_c, y_c = distortion_center[0], distortion_center[1]

    # --- initial guess: one-shot inverse (very important) ---
    r_d2 = (x_d - x_c)**2 + (y_d - y_c)**2
    scale_d = 1 + k1*r_d2 + k2*r_d2**2 + k3*r_d2**3

    x_u = x_d + (x_d - x_c)*(scale_d - 1)
    y_u = y_d + (y_d - y_c)*(scale_d - 1)

    # --- Newton iterations ---
    for _ in range(max_iter):
        dx = x_u - x_c
        dy = y_u - y_c

        r2 = dx*dx + dy*dy
        r4 = r2*r2
        r6 = r4*r2

        scale = 1 + k1*r2 + k2*r4 + k3*r6
        dscale_dr2 = k1 + 2*k2*r2 + 3*k3*r4

        # Forward distortion
        x_hat = x_u + dx*(scale - 1)
        y_hat = y_u + dy*(scale - 1)

        # Residual
        F = np.array([
            x_hat - x_d,
            y_hat - y_d
        ])

        if np.linalg.norm(F) < tol:
            break

        # Jacobian
        J11 = scale + 2*dx*dx*dscale_dr2
        J22 = scale + 2*dy*dy*dscale_dr2
        J12 = 2*dx*dy*dscale_dr2
        J21 = J12

        J = np.array([
            [J11, J12],
            [J21, J22]
        ])

        # Newton update
        delta = np.linalg.solve(J, F)
        x_u -= delta[0]
        y_u -= delta[1]

    return np.reshape(np.array([x_u, y_u]), (2,1))

def undistort_one_shot(point: Float[np.ndarray, "2 1"], distortion_center: Float[np.ndarray, "2 1"], k1: float, k2: float, k3: float, iterations = 4000) -> Float[np.ndarray, "2 1"]:
    """
    Performs distortion correction according to the Brown Conrady Model as a "one shot" calculation.
    """
    
    x_c = distortion_center[0]
    y_c = distortion_center[1]

    x_d = point[0]
    y_d = point[1]

    r = np.sqrt((x_d - x_c)**2 + (y_d - y_c)**2)
    f = lambda x_d, x_c, r: x_d + (x_d - x_c) * (k1 * (r ** 2) + k2 * (r ** 4) + k3 * (r ** 6))

    x_u = f(x_d, x_c, r)
    y_u = f(y_d, y_c, r)

    return np.reshape(np.array([x_u, y_u]), (2,1))

def undistort_one_shot_no_sqrt(point: Float[np.ndarray, "2 1"], distortion_center: Float[np.ndarray, "2 1"], k1: float, k2: float, k3: float, iterations = 4000) -> Float[np.ndarray, "2 1"]:
    """
    Performs distortion correction according to the Brown Conrady Model as a "one shot" calculation. Omits the square root for possible accuracy improvements
    """
    
    x_c = distortion_center[0]
    y_c = distortion_center[1]

    x_d = point[0]
    y_d = point[1]

    r = (x_d - x_c)**2 + (y_d - y_c)**2
    f = lambda x_d, x_c, r: x_d + (x_d - x_c) * (k1 * (r) + k2 * (r ** 2) + k3 * (r ** 3))

    x_u = f(x_d, x_c, r)
    y_u = f(y_d, y_c, r)

    return np.reshape(np.array([x_u, y_u]), (2,1))

def undistort_iter_over(point: Float[np.ndarray, "2 1"], distortion_center: Float[np.ndarray, "2 1"], k1: float, k2: float, k3: float, iterations = 40) -> Float[np.ndarray, "2 1"]:
        """
        Performs distortion correction according to the Brown Conrady Model based on the "one shot" estimation formula.
        Recalculates r for some reason. Doesn't work. 
        """
        
        x_c = distortion_center[0]
        y_c = distortion_center[1]

        x_d = point[0]
        y_d = point[1]

        r = np.sqrt((x_d - x_c)**2 + (y_d - y_c)**2)
        f = lambda x_d, x_c, r: x_d + (x_d - x_c) * (k1 * (r ** 2) + k2 * (r ** 4) + k3 * (r ** 6))

        x_u = f(x_d, x_c, r)
        y_u = f(y_d, y_c, r)

        for _ in range(iterations):
            r = np.sqrt((x_u - x_c)**2 + (x_u - y_c)**2)
            x_u = f(x_d, x_c, r)
            y_u = f(y_u, y_c, r)

        return np.reshape(np.array([x_u, y_u]), (2,1))