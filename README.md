## CMA-ES with Learning Rate Adaptation: Can CMA-ES with Default Population Size Solve Multimodal and Noisy Problems?

### About
This repository contains the code for the paper
"[CMA-ES with Learning Rate Adaptation: Can CMA-ES with Default Population Size Solve Multimodal and Noisy Problems?](https://arxiv.org/abs/2304.03473)"
by Masahiro Nomura, Youhei Akimoto, and Isao Ono, which has been accepted to [GECCO'23](https://gecco-2023.sigevo.org/HomePage).


### Dependencies
* numpy=1.24.2


### Example (Noiseless)
In the noiseless case, e.g. Sphere function, you can run the code by the following command:
```bash
python main.py --function=sphere \
  --dim=10 \
  --mean=3.0 \
  --sigma=2.0
```
Users can specify the experimental settings by adding the following flags:
* --function: objective function (required; please see the below)
* --dim: # dimension (required)
* --mean: initial mean vector (required; currently only scalar value is accepted)
* --sigma: initial step-size (required)
* --max_evals: maximum # evaluations (default=10000000; int)
* --criterion: target value, i.e., the optimization will stop when the function value reaches it (default=1e-3; float)

You can run the experiments on other functions by specifying such arguments in the same way.
The benchmark functions include
`sphere` (Sphere), `ellipsoid` (Ellipsoid), `rosen` (Rosenbrock),
`ackley` (Ackley), `schaffer` (Schaffer), `rastrigin` (Rastrigin), 
`rastrigin` (Rastrigin), `bohachevsky` (Bohachevsky), and `griewank` (Griewank),


### Example (Noisy)
In the noiseless case, e.g. NoisySphere function with variance=1.0, you can run the code by the following command:

```bash
python main.py --function=noisysphere-var=1.0 \
  --dim=10 \
  --mean=3.0 \
  --sigma=2.0
```
You can specify the variance by changing the `1.0` in the `noisysphere-var=1.0`.
The benchmark functions include
`noisysphere-var=...` (NoisySphere), `noisyellipsoid-var=...` (NoisyEllipsoid), and `noisyrastrigin-var=...` (NoisyRastrigin).
