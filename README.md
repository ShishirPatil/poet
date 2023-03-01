# POET

By Shishir G. Patil, Paras Jain, Prabal Dutta, Ion Stoica, and Joseph E. Gonzalez ([Project Website](https://poet.cs.berkeley.edu/))

![](https://github.com/ShishirPatil/poet/blob/gh-pages/assets/img/logo.png)

_See the paper!_ [https://arxiv.org/abs/2207.07697](https://arxiv.org/abs/2207.07697)

`POET` enables the training of state-of-the-art memory-hungry ML models on smartphones and other edge devices. POET (Private Optimal Energy Training) exploits the twin techniques of integrated tensor rematerialization, and paging-in/out of secondary storage \(as detailed in our paper at ICML 2022\) to optimize models for training with limited memory. POET's Mixed Integer Linear Formulation (MILP) ensures the solutions are provably optimal!

With POET, we are the first to demonstrate how to train memory-hungry SOTA ML models such as BERT and
ResNets on smartphones and tiny ARM Cortex-M devices :muscle:

Reach out to us at [sgp@berkeley.edu](mailto:sgp@berkeley.edu), if you have large models that you are trying to train - be it on GPUs, or your commodity edge devices such as laptops, smartphones, raspberry-pis, ARM Cortex M and A class, fitbits, etc.

## Get Started [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1iup_edJd9zB1tfVBHXLmkWOT5yoSmXzz?usp=sharing)

### Installation

Clone the repository and install POET:

```bash
git clone https://github.com/ShishirPatil/poet.git
cd poet/
pip install -e .
```

### Setting Up Gurobi (Recommended)

The ILP solver defaults to using the COIN-OR CBC solver when Gurobi isn't available. However, since Gurobi is much faster and presents solutions with tighter constraints, it is recommended to install it when possible.

#### Acquiring a Free Academic Gurobi Web License

If you are affiliated with an academic institution, you can acquire a free Gurobi Web License:

1. Create a free Gurobi account [here](https://pages.gurobi.com/registration). Make sure to specify the Academic user option.

<img width="602" alt="image" src="https://user-images.githubusercontent.com/52852612/206888332-cefa3d3e-9514-49f1-8bd1-82516a16ca08.png">

2. Complete the rest of the Gurobi account creation process, which will include creating a password and verifying your email address.
3. Login to the Gurobi [Web License Manager](https://license.gurobi.com/) using your new account.
4. Create and download a new Web License file. It will be called `gurobi.lic`.

<img width="493" alt="image" src="https://user-images.githubusercontent.com/52852612/206888423-4b3588bb-9724-4f38-96c3-778a8fff15af.png">

5. Move the `gurobi.lic` file to your home directory (i.e. to `~/gurobi.lic` on MacOS/Linux, or `C:\Users\YOUR_USERNAME\gurobi.lic` on Windows).

### Running the Solver via the Command Line

Once you have installed POET and optionally configured Gurobi, you can run the solver via the command line. Here's an example:

```bash
python poet/solve.py --model resnet18_cifar --platform a72 --ram-budget 3000000 --runtime-budget 7.6
```

### Using the Solver API Directly

If you'd like to use the solver API directly, you can do so as follows:

```python
from poet import solve

# ... use the solver API here
solve(
    model="resnet18_cifar",
    platform="m4",
    ram_budget=737719,
    runtime_budget=1.5,
    time_limit_s=400,
    solve_threads=16,
)
```

## Key ideas

From our [paper at ICML 2022](https://arxiv.org/abs/2207.07697):

```text
In this work, we show that paging and rematerialization are highly complementary.
By carefully rematerializing cheap operations while paging results of expensive operations
to auxiliary memory such as a flash or an SD card, we can scale effective memory capacity
with minimal energy over- head. By combining these two methods, we demonstrate it is
possible to train models like BERT on mobile-class edge devices. By framing edge training
as an optimization problem, we discover optimal schedules with provable minimal energy
consumption at a given memory budget. While the focus of this paper is edge deployments,
the energy objective is increasingly becoming relevant even for cloud deployments!
```

## Citation

If you use POET in your work, please cite us with:

```text
@inproceedings{patil2022poet,
  title={POET: Training Neural Networks on Tiny Devices with Integrated Rematerialization and Paging},
  author={Patil, Shishir G and Jain, Paras and Dutta, Prabal and Stoica, Ion and Gonzalez, Joseph},
  booktitle={International Conference on Machine Learning},
  pages={17573--17583},
  year={2022},
  organization={PMLR}
}
```
