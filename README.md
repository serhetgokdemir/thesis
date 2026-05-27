# Bachelor's Thesis Project in Mathematical Engineering

This repository contains the implementations and required documentation for a numerical root-finding methods project, developed as part of a Bachelor's thesis in the Department of Mathematical Engineering at Yildiz Technical University. The primary goal is to produce clean, testable, and mathematically sound code that adheres to academic standards.

Developed by Serhet Gökdemir under the supervision of Hale Gonce Köçken.

## Project Structure

The project is organized around a hierarchy that includes the LaTeX source for the thesis, Python implementations of numerical methods, and experiment/result files. The structure is as follows:

```
thesis-nonlinear-equations/
│
├── README.md
├── requirements.txt
├── pytest.ini
│
├── thesis/
│   ├── main.tex
│   ├── chapters/
│   ├── frontmatter/
│   ├── figures/
│   ├── tables/
│   ├── bibliography/
│   └── output/
│
├── src/
│   ├── single_variable/
│   │   ├── bisection.py
│   │   ├── secant.py
│   │   ├── newton.py
│   │   ├── damped_newton.py
│   │   ├── adaptive_damped_newton.py
│   │   ├── brent.py
│   │   └── __init__.py
│   │
│   ├── systems/
│   │   ├── newton_system.py
│   │   ├── broyden.py
│   │   └── __init__.py
│   │
│   └── __init__.py
│
├── tests/
│   ├── test_bisection.py
│   ├── test_secant.py
│   ├── test_newton.py
│   ├── test_damped_newton.py
│   ├── test_adaptive_damped_newton.py
│   ├── test_brent.py
│   ├── test_newton_system.py
│   └── test_broyden.py
│
├── experiments/
│   ├── single_variables_experiment.ipynb
│   ├── systems_experiments.ipynb
│   ├── armijo_experiments.ipynb
│   └── outputs/
│       ├── figures/
│       └── tables/
│
├── checkpoints/  # Contains historical progress PDFs (e.g., 03.17.2026.pdf)
└── forbidden/    # Personal or non-repository files (ignored by .gitignore)
```

## Implemented Methods

The project includes the following numerical methods for finding roots of nonlinear equations:

### Single-Variable Equations
- **Bisection Method**: A simple bracketing method.
- **Secant Method**: An open method that uses a succession of roots of secant lines.
- **Newton's Method**: An open method that uses tangent lines.
- **Damped Newton's Method**: A variation of Newton's method with a fixed damping factor.
- **Adaptive Damped Newton's Method**: An extension of Newton's method where the damping factor is chosen adaptively using the Armijo rule to ensure convergence.
- **Brent's Method**: A hybrid root-finding algorithm combining the bisection method, the secant method, and inverse quadratic interpolation.

### Systems of Nonlinear Equations
- **Newton's Method for Systems**: An extension of Newton's method for multi-dimensional problems.
- **Broyden's Method**: A quasi-Newton method for finding roots in multiple dimensions.

## Installation and Usage

You can run the project on your local machine by following the steps below.

### 1. Create a Virtual Environment (Recommended)

It is recommended to create a virtual environment to isolate project dependencies from your system.

```bash
python3 -m venv venv
source venv/bin/activate
```
On Windows, use `venv\Scripts\activate`.

### 2. Install Dependencies

The libraries required for the project are listed in the `requirements.txt` file. You can install these dependencies using `pip`.

```bash
pip install -r requirements.txt
```

### 3. Running Tests

The project has a suite of tests to ensure the correctness of the implemented methods. You can run the tests using `pytest`.

```bash
pytest
```

### 4. Running Experiments

The `experiments/` directory contains Jupyter notebooks for experimenting with the implemented methods and generating results. You can run the notebooks using Jupyter Lab.

```bash
jupyter lab
```

## Thesis Document

The thesis is written in LaTeX and can be compiled from the `latex/` directory. The `latexmk` tool is used for compilation, which automates the process of generating a PDF from the source files.

### Compiling the Thesis

To compile the thesis and generate a PDF, navigate to the `latex/` directory and run the following command:

```bash
cd latex
latexmk -pdf -outdir=compiled main.tex
```

### Cleaning Up Compilation Files

After compilation, you may want to clean up the generated files. To do this, you can run the following command from the `latex/` directory:

```bash
latexmk -C -outdir=compiled main.tex
```

