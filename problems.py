import numpy as np


def sphere(x):
    return np.sum(x**2)


class NoisySphere:
    def __init__(self, dim, seed, noisevar):
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        self.noisevar = noisevar

    def __call__(self, x):
        return sphere(x) + np.random.normal(0.0, np.sqrt(self.noisevar))

    def truefval(self, x):
        return sphere(x)


def ellipsoid(x):
    n = len(x)
    if len(x) < 2:
        raise ValueError("dimension must be greater one")
    Dell = np.diag([10 ** (3 * i / (n - 1)) for i in range(n)])
    return sphere(Dell @ x)


class NoisyEllipsoid:
    def __init__(self, dim, seed, noisevar):
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        self.noisevar = noisevar

    def __call__(self, x):
        return ellipsoid(x) + np.random.normal(0.0, np.sqrt(self.noisevar))

    def truefval(self, x):
        return ellipsoid(x)


def rosenbrockchain(x):
    if len(x) < 2:
        raise ValueError("dimension must be greater one")
    # return np.sum([100*(x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(self.n-1)])
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def rastrigin(x):
    n = len(x)
    if n < 2:
        raise ValueError("dimension must be greater one")
    return 10 * n + sum(x**2 - 10 * np.cos(2 * np.pi * x))


class NoisyRastrigin:
    def __init__(self, dim, seed, noisevar):
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        self.noisevar = noisevar

    def __call__(self, x):
        return rastrigin(x)[0] + np.random.normal(0.0, np.sqrt(self.noisevar))

    def truefval(self, x):
        return rastrigin(x)[0]


def ackley(x):
    n = len(x)
    f_value = 20.0
    tmp1 = np.sum([pow(x[i], 2.0) for i in range(n)])
    tmp2 = np.sum([np.cos(2.0 * np.pi * x[i]) for i in range(n)])
    f_value -= 20.0 * np.exp(-0.2 * np.sqrt(tmp1 / n))
    f_value += np.exp(1.0)
    f_value -= np.exp(tmp2 / n)
    # additional bound
    # see: Hansen(PPSN'04), Evaluating the CMA Evolution Strategy on Multimodal Test Functions
    bound_deviation = np.array([np.abs(x[i]) - 30 for i in range(n)])
    bound_deviation = np.where(bound_deviation > 0.0, 1.0, 0.0)
    penalty = (10**4) * np.sum(bound_deviation * np.square(x))
    return f_value + penalty


def bohachevsky(x):
    n = len(x)
    f_value = 0.0
    for i in range(n - 1):
        f_value += pow(x[i], 2.0)
        f_value += 2 * pow(x[i + 1], 2.0)
        f_value -= 0.3 * np.cos(3 * np.pi * x[i])
        f_value -= 0.4 * np.cos(4 * np.pi * x[i + 1])
        f_value += 0.7

    return f_value


def schaffer(x):
    n = len(x)
    return np.sum(
        [
            pow(pow(x[i], 2.0) + pow(x[i + 1], 2.0), 0.25)
            * (
                pow(np.sin(50 * pow(pow(x[i], 2.0) + pow(x[i + 1], 2.0), 0.1)), 2.0)
                + 1.0
            )
            for i in range(n - 1)
        ]
    )


def griewank(x):
    n = len(x)
    term_one = np.sum(x**2) / 4000.0
    term_two = np.prod([np.cos(x[i] / np.sqrt(i + 1)) for i in range(n)])
    return term_one - term_two + 1.0


def get_problem(problem, dim=None, seed=None, noisevar=None):
    if problem == "sphere":
        obj_func = sphere
    elif problem == "ellipsoid":
        obj_func = ellipsoid
    elif problem == "rosen":
        obj_func = rosenbrockchain
    elif problem == "rastrigin":
        obj_func = rastrigin
    elif problem == "ackley":
        obj_func = ackley
    elif problem == "bohachevsky":
        obj_func = bohachevsky
    elif problem == "schaffer":
        obj_func = schaffer
    elif problem == "griewank":
        obj_func = griewank
    elif problem == "noisysphere":
        assert dim is not None
        assert seed is not None
        assert noisevar is not None
        obj_func = NoisySphere(dim=dim, seed=seed, noisevar=noisevar)
    elif problem == "noisyellipsoid":
        assert dim is not None
        assert seed is not None
        assert noisevar is not None
        obj_func = NoisyEllipsoid(dim=dim, seed=seed, noisevar=noisevar)
    elif problem == "noisyrastrigin":
        assert dim is not None
        assert seed is not None
        assert noisevar is not None
        obj_func = NoisyRastrigin(dim=dim, seed=seed, noisevar=noisevar)
    else:
        raise NotImplementedError

    return obj_func
