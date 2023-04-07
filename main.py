import enum
import math
import sys
import numpy as np
import argparse

from problems import get_problem


class Solution(object):
    def __init__(self, dim):
        self.f = float("nan")
        self.x = np.zeros([dim, 1])
        self.z = np.zeros([dim, 1])


class LRType(enum.Enum):
    ADAPTIVE = "adaptive"
    FIXED = "fixed"

    def __repr__(self) -> str:
        return str(self)


def main(**params):
    function = params["function"]
    dim = params["dim"]
    mean = params["mean"]
    sigma = params["sigma"]
    seed = params["seed"]
    max_evals = int(params["max_evals"])
    criterion = params["criterion"]
    if function.startswith("noisy"):  # e.g. function = noisysphere-var=1.0
        function, noisevar = function.split("-")
        noisevar = float(noisevar.split("=")[1])
    else:
        noisevar = None
    obj_func = get_problem(function, dim=dim, seed=seed, noisevar=noisevar)
    lamb = 4 + int(3 * np.log(dim))  # default population size

    np.random.seed(seed)

    # constant
    mu = int(lamb / 2)
    wrh = math.log((lamb + 1.0) / 2.0) - np.log(np.arange(1, mu + 1))
    w = wrh / sum(wrh)
    mueff = 1 / np.sum(w**2, axis=0)
    cm = 1.0
    cs = (mueff + 2.0) / (dim + mueff + 5.0)
    ds = 1 + cs + 2 * max(0.0, np.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0)
    cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
    alpha_cov = min(2.0, lamb / 3.0)
    c1 = alpha_cov / (math.pow(dim + 1.3, 2) + mueff)
    cmu = min(
        1.0 - c1,
        alpha_cov
        * (mueff - 2.0 + 1.0 / mueff)
        / ((dim + 2.0) ** 2 + alpha_cov * mueff / 2.0),
    )
    chiN = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))
    I = np.eye(dim, dtype=float)

    # for LRA
    alpha = 1.4
    beta_mean = 0.1
    beta_Sigma = 0.03
    gamma = 0.1

    # variable
    mean = np.array([mean] * dim).reshape(dim, 1)
    C = np.eye(dim, dtype=float)
    ps = np.zeros([dim, 1])
    pc = np.zeros([dim, 1])
    evals = 0
    itr = 0
    sols = [Solution(dim) for _ in range(lamb)]
    fval = np.inf
    sqrtC = np.eye(dim, dtype=float)  # for sampling

    # for LRA
    Emean = np.zeros([dim, 1])
    ESigma = np.zeros([dim * dim, 1])
    Vmean = 0
    VSigma = 0
    eta_mean = 1.0
    eta_Sigma = 1.0

    try:
        while evals < max_evals:
            itr += 1

            nan_exists = False
            truefvals = []
            for i in range(lamb):
                sols[i].z = np.random.randn(dim, 1)
                sols[i].x = mean + sigma * sqrtC.dot(sols[i].z)
                sols[i].f = obj_func(sols[i].x)
                if sols[i].f is np.nan:
                    nan_exists = True
                if noisevar is not None:  # noisy case
                    truefvals.append(obj_func.truefval(sols[i].x))
            evals += lamb

            if nan_exists:
                print("NaN exists, so the optimization ends.")
                break

            sols = sorted(sols, key=lambda s: s.f)

            # fval is calculated at 'mean'
            if noisevar is None:
                fval = obj_func(mean)
            else:  # noisy case
                fval = obj_func.truefval(mean)

            print("#evals:{}, f(m):{}".format(evals, fval)) if itr % 1e4 == 0 else None

            if fval < criterion:
                break

            # keep old values
            old_mean = np.copy(mean)
            old_sigma = sigma
            old_Sigma = (sigma**2) * C
            e, v = np.linalg.eigh(C)  # remove the effect of sigma
            e = np.maximum(e, np.zeros_like(e))  # avoid domain error
            diag_invsqrt_eig = np.diag(1 / np.sqrt(e))
            old_inv_sqrtC = v @ diag_invsqrt_eig @ v.T

            wz = np.sum([w[i] * sols[i].z for i in range(mu)], axis=0)
            mean += cm * sigma * np.dot(sqrtC, wz)

            # evolution path
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * wz
            ps_norm = np.linalg.norm(ps)
            pc = (1.0 - cc) * pc + np.sqrt(cc * (2.0 - cc) * mueff) * np.dot(sqrtC, wz)

            # cov update
            if (ps_norm**2) / (1 - (1 - cs) ** (2 * (itr + 1))) < (
                2.0 + 4.0 / (dim + 1)
            ) * dim:
                hsig = 1
            else:
                hsig = 0
            rone = pc @ pc.T
            zrmu = np.sum(
                [w[i] * (np.outer(sols[i].z, sols[i].z) - I) for i in range(mu)], axis=0
            )
            rmu = sqrtC @ zrmu @ sqrtC.T
            C *= 1 - c1 + (1 - hsig) * c1 * cc * (2.0 - cc)
            C += c1 * rone + cmu * rmu

            # sigma update
            relative_velocity = (ps_norm / chiN) - 1.0
            sigma *= math.exp(min(1.0, (cs / ds) * relative_velocity))

            # calculate one-step difference of the parameters
            Deltamean = mean - old_mean
            Sigma = (sigma**2) * C
            # note that we use here matrix representation instead of vec one
            DeltaSigma = Sigma - old_Sigma

            # local coordinate
            old_inv_sqrtSigma = old_inv_sqrtC / old_sigma
            locDeltamean = old_inv_sqrtSigma.dot(Deltamean)
            locDeltaSigma = (
                old_inv_sqrtSigma.dot(DeltaSigma.dot(old_inv_sqrtSigma))
            ).reshape(dim * dim, 1) / np.sqrt(2)

            # moving average E and V
            Emean = (1 - beta_mean) * Emean + beta_mean * locDeltamean
            ESigma = (1 - beta_Sigma) * ESigma + beta_Sigma * locDeltaSigma
            Vmean = (1 - beta_mean) * Vmean + beta_mean * (
                np.linalg.norm(locDeltamean) ** 2
            )
            VSigma = (1 - beta_Sigma) * VSigma + beta_Sigma * (
                np.linalg.norm(locDeltaSigma) ** 2
            )

            # estimate SNR
            sqnormEmean = np.linalg.norm(Emean) ** 2
            hatSNRmean = (sqnormEmean - (beta_mean / (2 - beta_mean)) * Vmean) / (
                Vmean - sqnormEmean
            )

            sqnormESigma = np.linalg.norm(ESigma) ** 2
            hatSNRSigma = (sqnormESigma - (beta_Sigma / (2 - beta_Sigma)) * VSigma) / (
                VSigma - sqnormESigma
            )

            # update learning rate
            before_eta_mean = eta_mean
            relativeSNRmean = np.clip((hatSNRmean / alpha / eta_mean) - 1, -1, 1)
            eta_mean = eta_mean * np.exp(
                min(gamma * eta_mean, beta_mean) * relativeSNRmean
            )
            relativeSNRSigma = np.clip((hatSNRSigma / alpha / eta_Sigma) - 1, -1, 1)
            eta_Sigma = eta_Sigma * np.exp(
                min(gamma * eta_Sigma, beta_Sigma) * relativeSNRSigma
            )
            # cap
            eta_mean = min(eta_mean, 1.0)
            eta_Sigma = min(eta_Sigma, 1.0)

            # update parameters
            mean = old_mean + eta_mean * Deltamean
            Sigma = old_Sigma + eta_Sigma * DeltaSigma

            # decompose Sigma to sigma and C
            eigs, _ = np.linalg.eigh(Sigma)
            logeigsum = sum([np.log(e) for e in eigs])
            sigma = np.exp(logeigsum / 2.0 / dim)

            if sigma == 0:
                print("sigma is 0, so the optimization ends. ")
                break
            C = Sigma / (sigma**2)

            # step-size correction
            sigma *= before_eta_mean / eta_mean

            # for sampling in next iteration
            e, v = np.linalg.eigh(C)
            e = np.maximum(e, np.zeros_like(e))  # avoid domain error
            diag_sqrt_eig = np.diag(np.sqrt(e))
            sqrtC = v @ diag_sqrt_eig @ v.T

    except KeyboardInterrupt:
        print("KeyboardInterrupt: system exit...")
        sys.exit(0)

    print(evals, fval, seed)
    return evals, fval


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--function",
        choices=[
            # noiseless benchmarks
            "sphere",
            "ellipsoid",
            "rosen",
            "ackley",
            "schaffer",
            "rastrigin",
            "bohachevsky",
            "griewank",
            # noisy benchmarks
            # you can specify the variance by changing the =... (currently, variance=1.0)
            "noisysphere-var=1.0",
            "noisyellipsoid-var=1.0",
            "noisyrastrigin-var=1.0",
        ],
    )
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--mean", type=float, required=True)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_evals", type=int, default=10000000)
    parser.add_argument("--criterion", type=float, default=1e-3)
    main(**vars(parser.parse_args()))
