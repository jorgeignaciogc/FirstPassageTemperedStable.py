from decimal import DivisionByZero
import numpy as np, math
import scipy as sp
#from scipy import integrate

######################################
# Auxiliary parameters and functions #
######################################

# Number of nodes in Gaussian quadrature
class Stable:
    n_nodes = 2**10
    gl_x, gl_w = np.polynomial.legendre.leggauss(n_nodes)
    def __init__(self, a: float, th: float, b: bool = True):
        self.a = a
        self.th = th
        if b:
            self.gj_x, self.gj_w = sp.special.roots_jacobi(Stable.n_nodes, -a, 0.)

class TemperedStable:
    n_nodes = 2**8
    gl_x, gl_w = np.polynomial.legendre.leggauss(n_nodes)
    def __init__(self, a: float, th: float, q: float):
        self.a = a
        self.th = th
        self.q = q

class TruncatedTemperedStable:
    n_nodes = 2**8
    gl_x, gl_w = np.polynomial.legendre.leggauss(n_nodes)
    def __init__(self, a: float, th: float, q: float, r: float):
        self.a = a
        self.th = th
        self.q = q
        self.r = r


def si(a: float, x: float):
    if x == 0.:
        return (1-a) * a ** (a/(1-a))
    else: 
        try:
            return np.sin((1-a)*np.pi*x) * np.sin(a*np.pi*x) ** (a/(1-a)) / np.sin(np.pi*x) ** (1/(1-a))
        except:
            return float('inf')

def si(a: float, x: np.ndarray):
    aux1 = np.sin((1-a)*np.pi*x)
    aux2 = np.power(np.sin(a*np.pi*x), (a/(1-a)))
    aux3 = np.power(np.sin(np.pi*x), 1/(1-a))
    B = (x == 0.) | (aux1 == 0.) | (aux2 == 0.)
    with np.errstate(all='ignore'):
        return np.where(B, (1-a) * a ** (a/(1-a)), np.where(aux3 == 0., float('inf'), np.divide(np.multiply(aux1, aux2), aux3)))

def Dsi(a: float, x: float):   
    if x == 0. :
        return 0. 
    else: 
        return np.pi*si(a,x)*(a**2/np.tan(a*np.pi*x) + (1-a)**2/np.tan((1-a)*np.pi*x) - 1/np.tan(np.pi*x)) / (1-a)

def D2si(a: float, x: float):   
    return np.pi**2*si(a,x)*((a**2/np.tan(a*np.pi*x) + (1-a)**2/np.tan((1-a)*np.pi*x) - 1/np.tan(np.pi*x))**2 / (1-a)**2 + (- a**3/np.sin(a*np.pi*x)**2 - (1-a)**3/np.sin((1-a)*np.pi*x)**2 + 1/np.sin(np.pi*x)**2) / (1-a))

def DlogDsi(a: float, x: float):   
    aux = a**2/np.tan(a*np.pi*x) + (1-a)**2/np.tan((1-a)*np.pi*x) - 1/np.tan(np.pi*x)
    return np.pi*aux/(1-a) + (- a**3/np.sin(a*np.pi*x)**2 - (1-a)**3/np.sin((1-a)*np.pi*x)**2 + 1/np.sin(np.pi*x)**2) / aux

def Dlogsi(a: float, x: float): 
    if x == 0.:
        return  0. 
    else: 
        return np.pi*(a**2/np.tan(a*np.pi*x) + (1-a)**2/np.tan((1-a)*np.pi*x) - 1/np.tan(np.pi*x)) / (1-a)

def D2logsi(a: float, x: float):   
    return np.pi**2 * (1/np.sin(np.pi*x)**2 - a**3/np.sin(a*np.pi*x)**2 - (1-a)**3/np.sin((1-a)*np.pi*x)**2) / (1-a)

def D2si0(a: float, x: float):
    return np.pi**2*si0(a,x)*(a**2/np.tan(a*np.pi*x) + (1-a)**2/np.tan((1-a)*np.pi*x) - 1/np.tan(np.pi*x))**2 + np.pi**2*si0(a,x)*(1/np.sin(np.pi*x)**2 - a**3/np.sin(a*np.pi*x)**2 - (1-a)**3/np.sin((1-a)*np.pi*x)**2)

def si0(a: float, x: float):
    return np.sin((1 - a)*np.pi*x)**(1-a) * np.sin(a*np.pi*x)**a / np.sin(np.pi*x)

def Dsi0(a: float, x: float):   
    return np.pi*si0(a,x)*(a**2/np.tan(a*np.pi*x) + (1-a)**2/np.tan((1-a)*np.pi*x) - 1/np.tan(np.pi*x))

def Dlogsi0(a: float, x: float):   
    return np.pi*(a**2/np.tan(a*np.pi*x) + (1-a)**2/np.tan((1-a)*np.pi*x) - 1/np.tan(np.pi*x))

def D2si0(a: float, x: float):
    return np.pi**2*si0(a,x)*(a**2/np.tan(a*np.pi*x) + (1-a)**2/np.tan((1-a)*np.pi*x) - 1/np.tan(np.pi*x))**2 + np.pi**2*si0(a,x)*(1/np.sin(np.pi*x)**2 - a**3/np.sin(a*np.pi*x)**2 - (1-a)**3/np.sin((1-a)*np.pi*x)**2)

def Psi_aux(a: float, x: float):
    ca = 1-a
    ax = a*x
    cax = ca*x
    gax = math.gamma(1+ax)
    gcax = math.gamma(1+cax)
    ecax = math.exp(cax)
    eax = math.exp(ax-1)
    axx = ax ** ax
    caxx = cax ** cax
    sqax = math.sqrt(2*math.pi*a*cax)

    C1 = float('inf') if x == 0 else gax * eax * (a/ca+ax) ** (1+cax) / ax ** (x+1)
    C2 = gcax * ecax / caxx
    C3 = gax * eax * (1+1/cax) ** (1+cax) / (axx * sqax)
    C4 = gcax * ecax / (caxx *sqax)

    Cmin = min(C1,C2,C3,C4)

    if C1 == Cmin:
        return (1, math.exp(x)*gax*x ** (a/ca)/(ca*C1))
    elif C2 == Cmin:
        return (2, math.exp(x)*gcax/C2)
    elif C3 == Cmin: 
        return (3, math.erf(sqax * math.sqrt(math.pi)/2) * math.exp(x) * gax * x ** (a/ca) / (C3*sqax))
    else: # C4 == Cmin
        return (4, math.erf(sqax * math.sqrt(math.pi)/2) * math.exp(x)*gcax/(C4*sqax))

# Standardised PDF of S(1/th)
def phi(a: float, x: float):
    aux = si(a, (1 + Stable.gl_x) / 2)
    return (a/(2-2*a)) * x**(-1/(1-a)) * sum(Stable.gl_w * aux * np.exp(- aux * x**(-a/(1-a))))

def phi(a: float, x: np.ndarray):
    w = np.transpose(np.matrix(Stable.gl_w))
    aux = np.matrix(si(a, (1 + Stable.gl_x) / 2))
    aux2 = np.matmul(np.power(np.transpose(np.matrix(x)), -a/(1-a)), aux)
    return (a/(2-2*a)) * np.multiply(np.power(x,-1/(1-a)), np.array(np.transpose(np.matmul(np.multiply(aux, np.exp(-aux2)), w))[0]))

# PDF of S(t)
def pdf(S: Stable, t: float, x: float):
    sc = (S.th*t)**(-1/S.a)
    return phi(S.a, sc * x) * sc

def pdf(S: Stable, t: float, x: np.ndarray):
    sc = (S.th*t)**(-1/S.a)
    return phi(S.a, sc * x) * sc

# CDF of S(t)
def cdf(S: Stable, t: float, x: float):
    sc = (S.th*t)**(-1/S.a)
    aux = si(S.a, (1 + Stable.gl_x) / 2)
    return (1/2) * sum(Stable.gl_w * np.exp(- aux * (sc * x)**(-S.a/(1-S.a))))

def cdf(S: Stable, t: float, x: np.ndarray):
    sc = (S.th*t)**(-1/S.a)
    w = np.transpose(np.matrix(Stable.gl_w))
    aux = np.matrix(si(S.a, (1 + Stable.gl_x) / 2))
    aux = np.matmul(np.power(np.transpose(np.matrix(sc * x)), -S.a/(1-S.a)), aux)
    return (1/2) * np.array(np.transpose(np.matmul(np.exp(-aux), w)))[0]


# CDF of the overshoot S(τ_b)-b for constant b>0
def cdf_overshoot(S: Stable, b: float, x: float):
    return 1 - sp.special.betainc(S.a, 1-S.a, b/(x+b))

# CDF of the undershoot S(τ_b-)|τ_b=t for constant b>0
def cdf_undershoot(S: Stable, t: float, b: float, x: float):
    w = np.transpose(np.matrix(S.gj_w))
    aux = np.matrix(pdf(S,t, (S.gj_x + 1) * (b/2)))
    aux = np.matmul(np.where(S.gj_x > (2*x/b-1), aux, 0.), w)[0,0]
    return 1 - aux * ((2/b)**(S.a-1) * S.th * t * S.a / (sp.special.gamma(1-S.a) * pdf(S,t,b)[0]))

def cdf_undershoot(S: Stable, t: float, b: float, x: np.ndarray):
    w = np.transpose(np.matrix(S.gj_w))
    aux = np.matrix(pdf(S,t, (S.gj_x + 1) * (b/2)))
    aux = np.array(np.transpose(np.matmul(np.where(S.gj_x > np.transpose(np.matrix(2*x/b-1)), aux, 0.), w)))[0]
    return 1 - aux * ((2/b)**(S.a-1) * S.th * t * S.a / (sp.special.gamma(1-S.a) * pdf(S,t,b)[0]))


# f = np.logconcave density on [0,L], 1≥L, with mode at 0 and f(0) = 1
# a = largest 2**{-i}, i≥1, such that f(a) > 1/4
# b = f(a) > 1/4 ≥ f(2*a) = c
def Devroye_logconcave(f: 'function', a: float, b: float, c: float, L: float):
    if c != 0:
        l1 = np.log(b)
        l2 = np.log(c)
        dl = l1-l2
        s = 1/(1 + b + c/dl)
        X = 0
        while True:
            U = np.random.rand()
            if U < s:
                X = np.random.rand() * a
            elif U < (1+b)*s:
                X = a * (1 + np.random.rand())
            else:
                X = a * (2 - np.log(np.random.rand()) / dl)
            if X < L and np.random.rand() < f(X) / ( 1. if X < a else b if X < 2*a else np.exp(((2*a-X)*l1 + (X-a)*l2)/a)):
                return X
    else:
        l1 = np.log(b)
        s = 1/(1 + b)
        X = 0
        while True:
            U = np.random.rand()
            if U < s:
                X = np.random.rand() * a
            else:
                X = a * (1 + np.random.rand())
            if np.random.rand() < f(X) / ( 1 if X < a else b ):
                return X

# Inverse of the si function
def invsi(a: float, x: float):
    s0 = si(a, 0.)
    if x <= s0:
        return 0.
    n = 8
    N = 100
    s = np.log(x) * (1-a)
    def ssi(x: float): 
        return a*np.log(np.sin(a*np.pi*x)) + (1-a)*np.log(np.sin((1-a)*np.pi*x)) - np.log(np.sin(np.pi*x)) - s
    def Dssi(x: float):
        return np.pi*(a**2/np.tan(a*np.pi*x) + (1-a)**2/np.tan((1-a)*np.pi*x) - 1/np.tan(np.pi*x))
    # Binary search
    M = 3/(np.pi*(1-a**3-(1-a)**3))
    delta = 1/2
    y = 0
    i = 0
    while True:
        i += 1
        if ssi(y+delta) < 0:
            y += delta
        if i >= N or (y + delta < 1 and M * delta < y*(y+delta)**2*(1-y-delta)):
            y += delta
            break
        delta /= 2
    # Newton-Raphson
    for i in range(n):
        z = y - ssi(y) / Dssi(y) # Newton
        if z == y:
            return z
        elif z > 1 or z < 0:
            z = y - ssi(y)
            if z > 1:
                z = (y+1)/2
            elif z < 0:
                z = y/2
        y = z
    return y

# Inverse of Si: u ↦ usi(u)**a on [0,z]
def invSi(a,z,y):
    n = 8
    N = 100
    # Binary search
    x0 = z
    delta = z/2
    i = 0
    while True:
        i += 1
        if (x0-delta)*si(a,x0-delta)**a > y:
            x0 -= delta
        aux1 = Dlogsi(a, x0)
        aux2 = D2logsi(a, x0)
        aux3 = Dlogsi(a, x0-delta)
        if i >= N or delta * (1 + x0 * a * (aux2 + a * aux1**2)/2) < 1 + max(x0-delta,0) * a * aux3:
            break
        delta /= 2
    # Newton-Raphson
    for i in range(n):
        y0 = x0 - (x0 - y / si(a,x0)**a) / (1 + x0*a*Dlogsi(a,x0))
        if y0 == x0:
            return x0
        x0 = y0
    return x0

# Simulation from the PDF proportional to the positive def u ↦ (1-u)**(-r)*(a+b(1-u)) on [1/2,z]
def rand_neg_pow(r: float, a: float, b: float, z: float):
    if r != 2:
        a2, b2 = a/(1-r), b/(2-r)
        C0 = a2 * 2**(r-1) + b2 * 2**(r-2)
        C = a2 * (1-z)**(1-r) + b2 * (1-z) - C0
        # Newton-Raphson
        x0 = z
        y = np.random.rand()*C + C0
        n = 100
        for i in range(n):
            aux = 1-x0
            y0 = x0 + (a2*aux**(1-r) + b2*aux**(2-r) - y)/(a*aux**(-r) + b*aux**(1-r)) 
            if y0 == x0:
                return x0
            x0 = y0
        return x0
    else:
        C0 = 2*a + b * np.log(2)
        C = a / (1-z) - b * np.log(1-z) - C0
        # Newton-Raphson
        x0 = z
        y = np.random.rand() * C + C0
        n = 100
        for i in range(n):
            aux = 1-x0
            y0 = x0 - (a/aux - b*np.log(aux) - y)/(a/aux**2 + b/aux) 
            if y0 == x0:
                return x0
            x0 = y0
        return x0

# Integral x ↦ ∫_0**x si(u)**a*np.exp(-si(u)s)du for s≥0 and x∈[0,1]
def int_tilt_exp_si(a: float, s: float, x: float):
    z = invsi(a,a/s)
    return int_tilt_exp_si(a, s, x, z)

# If precomputed, z should be the critical point: z = invsi(a/s)
def int_tilt_exp_si(a: float, s: float, x: float, z: float):
    with np.errstate(all='ignore'):
        if 0. < z and z < x:
            # Break the integral in two: [0,z] and [z,x]
            x1 = (Stable.gl_x + 1) * (z/2)
            six1 = si(a, x1)
            six1 = np.where(six1 == float('inf'), 0., six1)
            x2 = (x+z)/2 + Stable.gl_x * (x-z)/2
            six2 = si(a, x2)
            six2 = np.where(six2 == float('inf'), 0., six2)
            return sum(np.multiply(Stable.gl_w, (np.multiply(np.power(six1, a), np.exp(- six1 * s))))) * z/2 + sum(np.multiply(Stable.gl_w, (np.multiply(np.power(six2, a), np.exp(- six2 * s))))) * (x-z)/2
        else:
            x1 = (Stable.gl_x + 1) * (x/2)
            six1 = si(a, x1)
            six1 = np.where(six1 == float('inf'), 0., six1)
            return sum(np.multiply(Stable.gl_w, np.multiply(np.power(six1, a), np.exp(- six1 * s)))) * x/2

# Integral x ↦ ∫_0**x np.exp(-si(u)s)du for s≥0 and x∈[0,1]
def int_exp_si(a: float, s: float, x: float):
    with np.errstate(all='ignore'):
        x1 = (Stable.gl_x + 1) * (x/2)
        y = si(a, x1)
        return sum(Stable.gl_w * np.exp(- y * s)) * x/2

# Integral x ↦ ∫_0**x si(u)**a du for x∈[0,1]
def int_pow_si(a: float, x: float):
    return sum(Stable.gl_w * si(a, (Stable.gl_x + 1) * (x/2)) ** a) * x/2

# Simulation from the density proportional to
# u ↦ np.exp(-si(u)s) on the interval [z,1] for some z∈(0,1)
def rand_exp_si(a: float, s: float, z: float):
    aux0 = si(a,z)
    def f(x: float):
        return np.exp(s * ( aux0 - si(a, z+x)))
    aux = np.log(4) / s + aux0
    a1 = (1-z)/2
    while si(a, z + a1) > aux:
        a1 /= 2
    a2 = f(a1)
    U = 0
    if a1 == (1-z)/2:
        a4 = 1 / (1 + a2)
        while True:
            V = np.random.rand(3)
            U = a1 * V[0] if V[1] < a4 else a1 * (1+V[0])
            if V[2] < f(U) / ( 1 if U < a1 else a2):
                return U
    else:
        a3 = f(2*a1)
        a4 = 1 / (1 + a2 + a3 / np.log(a2/a3))
        while True:
            V = np.random.rand(3)
            U = a1 * V[0] if V[1] < a4 else (a1 * (1+V[0]) if V[1] < a4 * (1+a2) else a1 * (2 + np.log(1/V[0])/np.log(a2/a3)))
            if U < 1 :
                if V[2] < f(U) / (1 if U < a1 else (a2 if U < 2*a1 else np.exp(((2*a1-U)*np.log(a2) + (U-a1)*np.log(a3))/a1))):
                    return U

# Simulation from the density proportional to
# u ↦ np.exp(-si(u)s) on the interval [0,1]
def rand_exp_si(a: float, s: float):
    aux0 = si(a,0)
    def f(x: float): 
        return np.exp(s * ( aux0 - si(a, x))) if (0 < x and x < 1) else 0.
    aux = np.log(4) / s + aux0
    a1 = 1/2
    while si(a, a1) > aux:
        a1 /= 2
    a2 = f(a1)
    a3 = 0. if a1 == 1/2 else f(2*a1)
    #println("Devroye for np.exp(-si(u)s)")
    return Devroye_logconcave(f,a1,a2,a3,1.)

# Simulation from the density proportional to
# u ↦ si(u)**a*np.exp(-si(u)s) on [0,1]
def rand_tilt_exp_si(a: float, s: float):
    z = invsi(a, a/s)
    return rand_tilt_exp_si(a, s, z)

# If precomputed, z = invsi(a, a/s)
def rand_tilt_exp_si(a: float, s: float, z: float):
    r = a / (1-a)
    def f0(u: float):
        auxf = si(a, u)
        return auxf ** a * np.exp(auxf * (-s))
    p = int_tilt_exp_si(a, s, 1., z)
    be = min(z,1/2)
    U = np.random.rand()
    if U > (int_tilt_exp_si(a, s, z, z) / p): # Sample lies on [z,1], where f is log-concave
        #print("Devroye for si(u)**a*np.exp(-si(u)s)")
        a1 = (1-z)/2
        aux = 1/f0(z) # (s/a) ** a * np.exp(a)
        def g(x) :
            return f0(z + x) * aux
        while g(a1) <= 1/4:
            a1 /= 2
        return z + Devroye_logconcave(g, a1, g(a1), 0. if a1 == (1-z)/2 else g(2*a1), 1-z)
    elif z <= 1/2 or U < (int_tilt_exp_si(a, s, be, z) / p): # Sample lies on [0,be]. Here we sample from Si: u ↦ usi(u)**a
        #print("Sample from u ↦ u*si(u)**a on [0,be]")
        C = be*si(a,be)**a
        while True:
            U = invSi(a, be, np.random.rand()*C)
            if np.random.rand() < (np.exp(-si(a,U)*s) / (1 + U * a * Dlogsi(a,U))):
                return U
    elif a <= 1/2: # Sample lies on [1/2,z]
        #print("Sample from u ↦ (1-u)**(-r)*(a+b(1-u)) on [z,1]")
        a1 = np.sin(np.pi*(1-a))
        b1 = np.pi*a*(1-a)*np.cos(np.pi*a)
        sc = 2**r * a**(1-a)
        while True:
            U = rand_neg_pow(r,a1,b1,z)
            aux = si(a,U)
            if np.random.rand() < (aux**a * np.exp(-s*aux) * (1-U)**r * sc / (a1 + b1*(1-U))) :
                return U
    else: # a>1/2, r>1, z>1/2 and sample lies on [1/2,z]
        #print("Sample from u ↦ ρ(u)**(r-1) on [1/2,z]")
        C0 = si0(a,1/2)**(r-1)
        C = si0(a,z)**(r-1) - C0
        c = Dlogsi0(a,1/2) / si0(a,1/2)
        while True:
            U = np.random.rand() * C + C0
            U = invsi(a, U**(1/(2*a-1)))
            if np.random.rand() < (c*si0(a,U)*np.exp(-s*si(a,U))/Dlogsi0(a,U)):
                return U

# Auxiliary def that produces (b, Db, B)
# given the parameters of b : t ↦ min(a0 - a1 t, r)
def crossing_functions(a: float, a0: float, a1: float, r: float):
    def b(t: float): 
        return min(a0 - a1*t, r)
    if a0 < 0:
        def Db(t):
            return 0.
        def B(t):
            return 0.
        return (b, Db, B)
    elif a1 == 0: 
        def Db(t):
            return 0.
        def B(t):
            return (min(a0,r)/t) ** a 
        return (b, Db, B)
    else:
        aux = (a0-r)/a1
        if aux > 0:
            def Db(t): 
                return -a1 if t > aux else 0
            def B(t):
                if t > r * aux**(-1/a) :
                    return (t/r)**(-a)
                else:
                    ra = 1/a
                    x = (a0 - t * aux**ra) / a1
                    for i in range(50):
                        y = x - (t * x**ra + a1 * x - a0) / (a1 + ra * t * x**(ra-1))
                        if y == x:
                            return x
                        else:
                            x = y
                    return x
            return (b, Db, B)
        else:
            def Db(t): 
                return -a1
            def B(t):
                ra = 1/a
                x = (t/a0 + (a1/a0)**ra)**(-a)
                for i in range(50):
                    y = x - (t * x**ra + a1 * x - a0) / (a1 + ra * t * x**(ra-1))
                    if y == x:
                        return x
                    else:
                        x = y
                return x
            return (b, Db, B)

###########################
# Main simulation methods #
###########################

# Sample of S(t) under P_0
def rand_stable(S: Stable, t: float):
    return (S.th*t)**(1/S.a) * (si(S.a, np.random.rand()) / (-np.log(np.random.rand())))**((1-S.a)/S.a)

# Sample of S(t) under ℙ_q
def rand_tempered_stable(S: TemperedStable, t: float):
    L = (S.th*t) ** (1/S.a)*S.q
    xi = L ** S.a
    r = S.a / (1-S.a)
    (i,Cmin) = Psi_aux(S.a,xi)
    if i == 1:
        ax = S.a * xi
        aux = xi ** (r+1)
        while True:
            U = np.random.rand()
            V = np.random.rand()
            X = np.random.gamma(ax, scale=1.0)
            rho = si0(S.a,U) ** (r+1)
            X1 = X ** (-r)
            if V <= rho * X ** (-ax) * X1 * math.exp(-rho*aux*X1) * Cmin:
                return X/S.q
    elif i == 2:
        cax = (1-S.a)*xi
        aux = xi ** (r+1)
        while True:
            U = np.random.rand()
            V = np.random.rand()
            X = np.random.gamma(1+cax, scale=1.0)
            s = si0(S.a,U) ** (1/S.a)*X ** (-1/r)
            if V <= X ** (-cax) * math.exp(-L*s) * Cmin:
                return (S.th * t) ** (1/S.a)*s
    elif i == 3:
        ax = S.a * xi
        aux = xi ** (r+1)
        sc = math.pi*math.sqrt((1-S.a)*ax/2)
        while True:
            U = np.random.rand()
            V = np.random.rand()
            U = sp.special.erfinv(U * math.erf(sc)) / sc
            X = np.random.gamma(ax,scale=1.0)
            rho = si0(S.a,U) ** (r+1)
            X1 = X ** (-r)
            if V <= rho * X ** (-ax) * X1 * math.exp((1-S.a)*ax*U ** 2/2-rho*aux*X1) * Cmin:
                return X/S.q
    else:# i == 4
        cax = (1-S.a)*xi
        aux = xi ** (r+1)
        sc = math.pi * math.sqrt(S.a*cax/2)
        while True:
            U = np.random.rand()
            V = np.random.rand()
            U = sp.special.erfinv(U * math.erf(sc)) / sc
            X = np.random.gamma(1+cax, scale=1.0)
            s = si0(S.a,U) ** (1/S.a)*X ** (-1/r)
            if V <= X ** (-cax) * math.exp(S.a*cax*U ** 2/2-L*s) * Cmin:
                return (S.th*t) ** (1/S.a)*s

# Sample of S(t)|{S(t)≤s} under P_0
def rand_small_stable(S: Stable, t: float, s: float):
    s1 = (S.th * t / s**S.a)**(1/(1-S.a))
    aux0 = si(S.a,0)
    def f(x: float): 
        return np.exp(s1 * ( aux0 - si(S.a, x)))
    aux = np.log(4) / s1 + aux0
    a1 = 1/2
    while si(S.a, a1) > aux:
        a1 /= 2
    a2 = f(a1)
    U = 0
    if a1 == 1/2:
        a4 = 1 / (1 + a2)
        while True:
            V = np.random.rand(3)
            U = a1 * V[0] if V[1] < a4 else a1 * (1+V[0])
            if V[2] < f(U) / ( 1 if U < a1 else a2):
                break
    else:
        a3 = f(2*a1)
        a4 = 1 / (1 + a2 + a3 / np.log(a2/a3))
        while True:
            V = np.random.rand(3)
            U = a1 * V[0] if V[1] < a4 else ( a1 * (1+V[0]) if V[1] < a4 * (1+a2) else a1 * (2 + np.log(1/V[0])/np.log(a2/a3)))
            if U < 1: 
                if V[2] < f(U) / (1. if U < a1 else (a2 if U < 2*a1 else np.exp(((2*a1-U)*np.log(a2) + (U-a1)*np.log(a3))/a1))):
                    break
    return (S.th*t)**(1/S.a) * (-np.log(np.random.rand()) / si(S.a, U) + s1)**(-(1-S.a)/S.a)

# Sample of S(t)|{S(t)≤s} under P_q
def rand_small_tempered_stable(S: TemperedStable, t: float, s: float):
    if 1/2 > max(math.exp(-S.q*s),math.exp(-S.q ** S.a *S.th * t)): # = E_q[exp(q(S(t)-s))] ≥ P_q(S(t)>s) thus P_q(S(t)<s) > 1/2
        x = rand_tempered_stable(S.a, S.th, S.q, t)
        while x > s:
            x = rand_tempered_stable(S.a, S.th, S.q, t)
        return x
    else: # max(math.exp(-S.q*s),math.exp(-S.q ** S.a *S.th * t)) ≥ 1/2
        x = rand_small_stable(S.a, S.th, t, s)
        while x > -np.log(np.random.rand()) / S.q:
            x = rand_small_stable(S.a, S.th, t, s)
        # Acceptance probability = ℙ_0[S(t)< E/q | S(t)<s] ≥ max(exp(-qs), exp(-q ** α θ t)) ≥ 1/2
        return x

# Sample of the undershoot S(t-) where: 
# t is the crossing time and s = b(t) is the crossing level
def rand_undershoot_stable(S: Stable, t: float, s: float):
    r = S.a / (1-S.a)
    sc = (S.th * t)**(1/S.a)
    s1 = s / sc
    s2 = s1**(-r)
    z = invsi(S.a, S.a/s2)
    p = sp.special.gamma(1-S.a) * int_tilt_exp_si(S.a, s2, 1., z)
    p = (2-2**S.a)**(-S.a) * r**(-S.a) * s1**(S.a*r) *  int_exp_si(S.a, s2, 1.) / p
    p = 1/(1+1/p)
    c1 = (1-2**(S.a-1))**(-S.a) * s1**(-S.a)
    c2 = (2*r)**S.a * s2
    E = 0.
    while True:
        if np.random.rand() <= p:
            U = rand_exp_si(S.a, s2)
            E = -np.log(np.random.rand()) / si(S.a,U) + s2
        else:
            U = rand_tilt_exp_si(S.a, s2, z)
            E = np.random.gamma(1-S.a, 1.) / si(S.a,U) + s2
        if np.random.rand() < np.abs(s1 - E**(-1/r))**(-S.a) / (c1 + c2 * ( (E - s2)**(-S.a) if E > s2 else 0.)):
            return sc * E**(-1/r)

# Sample of the vector (t, S(t-), S(t)-S(t-)) under ℙ_0 where: 
# t is the crossing time across the boundary b
# b : ℝ → ℝ is the target/boundary function
# Db : ℝ → ℝ is the derivative of b
# B : ℝ → ℝ  is the inverse function of t ↦ t ** (-1/α)b(t)   
def rand_crossing_stable(S: Stable, b: 'function', Db: 'function', B: 'function'):
    s = rand_stable(S, 1.)
    T = B(s)
    w0 = -Db(T)
    U0 = b(T)
    if w0 != 0:
        w1 = U0/(S.a*T)
        if np.random.rand() <= w0 / (w0 + w1):
            return (T, U0, 0.)
    U = rand_undershoot_stable(S, T, U0)
    return (T, U, (U0-U) * np.random.rand()**(-1/S.a))

# Sample of the vector (t, S(t-), S(t)-S(t-))|{t≤T} under ℙ_0 where: 
# T>0 is some time horizon
# t is the crossing time across the boundary b
# b : ℝ → ℝ is the target/boundary function
# Db : ℝ → ℝ is the derivative of b
# B : ℝ → ℝ is the inverse function of t ↦ t ** (-1/α)b(t)  

def rand_crossing_small_stable(S:Stable, b: 'function', Db: 'function', B: 'function', T: float):
    s = rand_stable(S, 1.)
    aux = T ** (-1/S.a) * b(T)
    while s < aux:
        s = rand_stable(S, 1.)
    t = B(s)
    w0 = -Db(t)
    U0 = b(t)
    if w0 != 0:
        w1 = U0/(S.a*t)
        if np.random.rand() <= w0 / (w0 + w1):
            return (t, U0, 0.)
    U = rand_undershoot_stable(S, t, U0)
    return (t, U, (U0-U) * np.random.rand() ** (-1/S.a))

# Sample of the vector (t, S(t-), S(t)-S(t-)) under ℙ_q where: 
# t is the crossing time across the boundary b
# b is the target/boundary function
# Db is the derivative of b
# BB : ℝ^2 → {f : ℝ → ℝ} so that B = BB(x0,x1) : ℝ → ℝ is the inverse function of t ↦ t^(-1/α)(b(t+x0)-x1)  
def rand_crossing_tempered_stable(S: TemperedStable, b: 'function', Db: 'function', BB: 'function'):
    Tmax = (2. * S.q * b(0) + 1. - 2 **(-S.a))/((2 ** S.a - 1.)*S.q ** S.a * S.th)
    Tf = Uf = 0.
    s = rand_tempered_stable(S,Tmax)
    while s < b(Tmax+Tf) - Uf:
        s = rand_tempered_stable(S,Tmax)
        Tf = Tf + Tmax
        Uf = Uf + s
    def locb(x: float): 
        return b(x+Tf) - Uf
    def locDb(x: float): 
        return Db(x+Tf)
    locB = BB(Tf,Uf)

    S0 = Stable(S.a,S.th)
    (T,U,V) = rand_crossing_small_stable(S0,locb,locDb,locB,Tmax)
    s = rand_stable(S0,Tmax-T)
    while S.q*(s+U+V) > -np.log(np.random.rand()):
        (T, U, V) = rand_crossing_small_stable(S0,locb,locDb,locB,Tmax)
        s = rand_stable(S0,Tmax-T)
    return (Tf+T, Uf+U, V)

# Sample of the vector (t, S(t-), S(t)-S(t-)) under ℙ_q where: 
# t is the crossing time across the boundary b
# b : t ↦ min(a0 - a1*t, r) is the target/boundary function
def rand_crossing_tempered_stable(S: TemperedStable, a0: float, a1: float, r: float):
    if S.q == 0. :
        return rand_crossing_stable(Stable(S.a,S.th), a0, a1, r)

    R0 = (2 ** S.a - 1.) / (2. * S.q)
    Tf = Uf = V = 0
    def locb(t):
        return 0.
    
    def locDb(t): 
        return 0.
    
    def locB(t): 
        return 0. 
    
    S0 = Stable(S.a,S.th)
    
    while Uf + V <= min(a0 - a1*Tf, r):
        Uf += V
        R = R0 + Uf
        Tmax = (2*S.q*min(a0,r,R-Uf)+1-2 ** (-S.a))/((2 ** S.a-1)*S.q ** S.a * S.th)
        s = rand_tempered_stable(S,Tmax)
        while s < min(a0 - a1*(Tmax+Tf), r, R) - Uf:
            Tf += Tmax
            Uf += s
            s = rand_tempered_stable(S,Tmax)

        (locb,locDb,locB) = crossing_functions(S.a, a0-a1*Tf, a1, min(r,R) - Uf)
        (T,U,V) = rand_crossing_small_stable(S0,locb,locDb,locB,Tmax)
        s = rand_stable(S0,Tmax-T)

        while S.q*(s+U+V) > locb(Tmax) - math.log(np.random.rand()): 
            (T,U,V) = rand_crossing_small_stable(S0,locb,locDb,locB,Tmax)
            s = rand_stable(S0,Tmax-T)
        Tf += T
        Uf += U
    return (Tf, Uf, V)

# Sample of (τ,Z(τ-),Z(τ)-Z(τ-)) where Z = Z ** +-Z ** -
# At the stopped time τ = τ0∧T where τ0 is the crossing time τ0=inf{t>0:Z_t>r} across level r
# Z ** + has parameters (α1,θ1,q1) and Z ** - has parameters (α2,θ2,q2)
def rand_crossing_BV(S1: TemperedStable, S2: TemperedStable, T: float, r: float):
    tt = H = 0.
    b = r
    (t,u,v) = rand_crossing_tempered_stable(S1,b,0.,b)
    w = rand_tempered_stable(S2,t)
    while u + v - w < b:
        if tt + t >= T:
            v = rand_small_tempered_stable(S1,T-tt,b) - rand_tempered_stable(S2,T-tt)
            return (T,H+v,0.)
        tt += t
        H += v - w
        b -= v - w
        (t,u,v) = rand_crossing_tempered_stable(S1,b,0.,b)
        w = rand_tempered_stable(S2,t)
    if tt + t >= T:
        v = rand_small_tempered_stable(S1,T-tt,b) - rand_tempered_stable(S2,T-tt)
        return (T,H+v,0.)
    return (tt+t,H+u-w,v)