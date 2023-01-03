from typing import Dict, Literal, Optional
import numpy as np

from poet.architectures.bert import BERTBase
from poet.architectures.linear import make_linear_network
from poet.architectures.resnet import resnet18, resnet18_cifar, resnet50
from poet.architectures.vgg import vgg16
from poet.chipsets import M4F, MKR1000, JetsonTX2, RPi, RPiNoCache
from poet.poet_solver import POETSolver
from poet.poet_solver_gurobi import POETSolverGurobi
from poet.util import make_dfgraph_costs


def solve(
    model: Literal[
        "linear",
        "vgg16",
        "vgg16_cifar",
        "resnet18",
        "resnet50",
        "resnet18_cifar",
        "bert",
        "transformer",
    ],
    platform: Literal["m0", "a72", "a72nocache", "m4", "jetsontx2"],
    ram_budget: float,
    runtime_budget: float,
    paging: int = 1,
    remat: int = 1,
    mem_power_scale=1.0,
    batch_size=1,
    # use_actual_gurobi uses Gurobi optimizer with the model defined in Gurobi.
    use_actual_gurobi: Optional[bool] = True,
    # solver defines the model using PuLP and can swap in either cbc or Gurobi solver
    solver: Optional[Literal["gurobi", "cbc"]] = None,
    time_limit_s: float = 1e100,
    solve_threads: Optional[int] = None,
) -> Dict:
    """Solve a POET LP problem.
    :param model: The model to solve for.
    :param platform: The platform to solve for.
    :param ram_budget: The RAM budget in bytes.
    :param runtime_budget: The runtime budget in milliseconds.
    :param paging: Whether to enable paging.
    :param remat: Whether to enable rematerialization.
    :param mem_power_scale: A scaling factor for the memory power.
    :param batch_size: The batch size to use for the model.
    :param use_actual_gurobi: Whether to use the actual Gurobi solver over the PuLP wrapper.
    :param solver: The LP solver to use.
    :param time_limit_s: The time limit for solving in seconds.
    :param solve_threads: The number of threads to use for solving.
    """
    if platform == "m0":
        chipset = MKR1000
    elif platform == "a72":
        chipset = RPi
    elif platform == "a72nocache":
        chipset = RPiNoCache
    elif platform == "m4":
        chipset = M4F
    elif platform == "jetsontx2":
        chipset = JetsonTX2
    else:
        raise NotImplementedError()

    chipset["MEMORY_POWER"] *= mem_power_scale

    # make model
    if model == "linear":
        net = make_linear_network()
    elif model == "vgg16":
        net = vgg16(batch_size)
    elif model == "vgg16_cifar":
        net = vgg16(batch_size, 10, (3, 32, 32))
    elif model == "resnet18":
        net = resnet18(batch_size)
    elif model == "resnet50":
        net = resnet50(batch_size)
    elif model == "resnet18_cifar":
        net = resnet18_cifar(batch_size, 10, (3, 32, 32))
    elif model == "bert":
        net = BERTBase(SEQ_LEN=512, HIDDEN_DIM=768, I=64, HEADS=12, NUM_TRANSFORMER_BLOCKS=12)
    elif model == "transformer":
        net = BERTBase(SEQ_LEN=512, HIDDEN_DIM=768, I=64, HEADS=12, NUM_TRANSFORMER_BLOCKS=1)
    else:
        raise NotImplementedError()

    # build graph
    graph_costs = make_dfgraph_costs(net=net, device=chipset)
    (
        g,
        cpu_power_cost_vec_joule,
        pagein_power_cost_vec_joule,
        pageout_power_cost_vec_joule,
    ) = graph_costs

    print("CPU power cost:", cpu_power_cost_vec_joule)
    print("Page-in power cost:", pagein_power_cost_vec_joule)
    print("Page-out power cost:", pageout_power_cost_vec_joule)

    total_runtime = sum(g.cost_cpu.values())

    total_runtime = sum(g.cost_cpu.values())
    runtime_budget_ms = runtime_budget * total_runtime

    if use_actual_gurobi:
        solver = POETSolverGurobi(
            g,
            cpu_power_cost_vec_joule,
            pagein_power_cost_vec_joule,
            pageout_power_cost_vec_joule,
            ram_budget,
            runtime_budget_ms,
            paging,
            remat,
            time_limit_s=time_limit_s,
            solve_threads=solve_threads,
        )
    else:
        solver = POETSolver(
            g,
            cpu_power_cost_vec_joule=cpu_power_cost_vec_joule,
            pagein_power_cost_vec_joule=pagein_power_cost_vec_joule,
            pageout_power_cost_vec_joule=pageout_power_cost_vec_joule,
            ram_budget_bytes=ram_budget,
            runtime_budget_ms=runtime_budget_ms,
            paging=paging,
            remat=remat,
            time_limit_s=time_limit_s,
            solve_threads=solve_threads,
            solver=solver,
        )

    solution = solver.solve()

    if solution is not None and solution.feasible:
        cpu_cost_vec = np.asarray([g.cost_cpu[i] for i in range(g.size)])[np.newaxis, :].T
        """
        Making it compatible with poet.utils.checkmate.core.utils.scheduler.py
        Gurobi schedule is the optimizer schedule, which may not be accurate.
        So we generate our own schedule that is tf2 compatible
        """
        total_power_cost_page = np.sum(solution.Min @ pagein_power_cost_vec_joule + solution.Mout @ pageout_power_cost_vec_joule)
        total_power_cost_cpu = np.sum(solution.R @ cpu_power_cost_vec_joule)
        total_runtime = np.sum(solution.R @ cpu_cost_vec)
    else:
        total_power_cost_page, total_power_cost_cpu, total_runtime = None, None, None

    result = dict(
        ram_budget_bytes=ram_budget,
        runtime_budget_ms=runtime_budget_ms,
        paging=paging,
        remat=remat,
        integral=True,
        solution=solution,
        total_power_cost_page=total_power_cost_page,
        total_power_cost_cpu=total_power_cost_cpu,
        total_runtime=total_runtime,
        feasible=(solution is not None and solution.feasible),
    )

    return result
