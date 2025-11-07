## 🧩 Setup Instructions

If you have **conda** installed (Miniconda or Anaconda):

```bash
conda create -n poker-pypy -c conda-forge pypy python=3.9
```
```bash
conda activate poker-pypy
```
```bash
pypy3 -m pip install -U pip wheel setuptools
```
```bash
pypy3 -m pip install -e .
```
```bash
pypy3 -m pip install gymnasium clubs-gym tqdm numpy
```
```bash
conda install -c conda-forge pillow matplotlib
```
```bash
pypy3 main.py



