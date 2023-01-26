from argparse import ArgumentParser
from typing import Literal, Optional

import numpy as np

from poet import solve
from poet.poet_solver import POETSolver
from poet.poet_solver_gurobi import POETSolverGurobi
from poet.util import get_chipset_and_net, make_dfgraph_costs, plot_dfgraph, print_result, POETResult
from gurobipy import GRB, GurobiError


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
    solver: Literal["gurobipy", "pulp-gurobi", "pulp-cbc"] = "gurobipy",
    print_power_costs: bool = False,
    print_graph_info: bool = True,
    plot_directory: Optional[str] = None,
    time_limit_s: float = 1e100,
    solve_threads: Optional[int] = None,
):
    """Solve a POET LP problem.
    :param model: The model to solve for.
    :param platform: The platform to solve for.
    :param ram_budget: The RAM budget in bytes.
    :param runtime_budget: The runtime budget in milliseconds.
    :param paging: Whether to enable paging.
    :param remat: Whether to enable rematerialization.
    :param mem_power_scale: A scaling factor for the memory power.
    :param batch_size: The batch size to use for the model.
    :param solver: The LP solver to use.
    :param time_limit_s: The time limit for solving in seconds.
    :param solve_threads: The number of threads to use for solving.
    """
    chipset, net = get_chipset_and_net(
        platform=platform,
        model=model,
        batch_size=batch_size,
        mem_power_scale=mem_power_scale,
    )

    # build graph
    graph_costs = make_dfgraph_costs(net=net, device=chipset)
    (
        g,
        cpu_power_cost_vec_joule,
        pagein_power_cost_vec_joule,
        pageout_power_cost_vec_joule,
    ) = graph_costs

    if plot_directory is not None:
        plot_dfgraph(g, plot_directory)

    if print_power_costs:
        print("CPU power cost:", cpu_power_cost_vec_joule)
        print("Page-in power cost:", pagein_power_cost_vec_joule)
        print("Page-out power cost:", pageout_power_cost_vec_joule)

    total_runtime = sum(g.cost_cpu.values())
    runtime_budget_ms = runtime_budget * total_runtime
    total_ram = sum(g.cost_ram[i] for i in g.vfwd)

    if print_graph_info:
        print(f"Total runtime of graph (forward + backward): {total_runtime:.5f} milliseconds")
        print(f"Total RAM consumption of forward pass: {total_ram} bytes")

    solver_params = dict(
        g=g,
        cpu_power_cost_vec_joule=cpu_power_cost_vec_joule,
        pagein_power_cost_vec_joule=pagein_power_cost_vec_joule,
        pageout_power_cost_vec_joule=pageout_power_cost_vec_joule,
        ram_budget_bytes=ram_budget,
        runtime_budget_ms=runtime_budget_ms,
        paging=paging,
        remat=remat,
        time_limit_s=time_limit_s,
        solve_threads=solve_threads,
    )

    if solver == "gurobipy":
        solver = POETSolverGurobi(**solver_params)
    else:
        solver = POETSolver(**solver_params, solver=solver)

    try:
        solution = solver.solve()
    except GurobiError as e:
        if e.errno == GRB.Error.SIZE_LIMIT_EXCEEDED:
            print("A valid Gurobi license was not found; retrying with CBC solver")
            solver = POETSolver(**solver_params, solver="pulp-cbc")
            solution = solver.solve()
        else:
            raise e

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

    result = POETResult(
        ram_budget=ram_budget,
        runtime_budget_ms=runtime_budget_ms,
        paging=paging,
        remat=remat,
        total_power_cost_page=total_power_cost_page,
        total_power_cost_cpu=total_power_cost_cpu,
        total_runtime=total_runtime,
        feasible=solution.feasible,
        solution=solution,
    )

    return result


if __name__ == "__main__":
    parser = ArgumentParser(description="Solve a POET LP problem")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["vgg16", "vgg16_cifar", "resnet18", "resnet50", "resnet18_cifar", "bert", "transformer", "linear"],
    )
    parser.add_argument("--platform", type=str, required=True, choices=["m0", "a72", "a72nocache", "m4", "jetsontx2"])
    parser.add_argument("--ram-budget", type=int, required=True)
    parser.add_argument("--runtime-budget", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mem-power-scale", type=float, default=1.0)
    parser.add_argument("--paging", action="store_true", default=True)
    parser.add_argument("--remat", action="store_true", default=True)
    parser.add_argument("--time-limit-s", type=int, default=1e100)
    parser.add_argument("--solve-threads", type=int, default=4)
    parser.add_argument("--solver", type=str, default="gurobipy", choices=["gurobipy", "pulp-gurobi", "pulp-cbc"])
    parser.add_argument("--use-actual-gurobi", action="store_true", default=False)
    parser.add_argument("--print-power-costs", action="store_true", default=False)
    parser.add_argument("--print-graph-info", action="store_true", default=True)
    parser.add_argument("--plot-directory", type=str, default=None)
    args = parser.parse_args()

    result = solve(
        model=args.model,
        platform=args.platform,
        ram_budget=args.ram_budget,
        runtime_budget=args.runtime_budget,
        batch_size=args.batch_size,
        mem_power_scale=args.mem_power_scale,
        paging=args.paging,
        remat=args.remat,
        time_limit_s=args.time_limit_s,
        solve_threads=args.solve_threads,
        solver=args.solver,
        print_power_costs=args.print_power_costs,
        print_graph_info=args.print_graph_info,
        plot_directory=args.plot_directory,
    )

    print_result(result)
