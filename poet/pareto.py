import os
from concurrent.futures import ProcessPoolExecutor
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from poet import solve
from poet.util import get_chipset_and_net, get_net_costs, make_dfgraph_costs


def solve_wrapper(params):
    return solve(**params)


def pareto(
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
    # ram_budget: float,
    runtime_budget: float = 1.4,
    mem_power_scale=1.0,
    batch_size=1,
    ram_budget_samples: int = 20,
    solver: Literal["gurobipy", "pulp-gurobi", "pulp-cbc"] = "gurobipy",
    time_limit_s: float = 1e100,
    solve_threads: int = 4,
    total_threads: int = os.cpu_count(),
    filename: str = "pareto.png",
):
    plt.ion()

    chipset, net, ram_budget_start, ram_budget_end = get_chipset_and_net(
        platform=platform,
        model=model,
        batch_size=batch_size,
        mem_power_scale=mem_power_scale,
    )

    base_memory = max(get_net_costs(net=net, device=chipset)["memory_bytes"])
    print(
        base_memory,
        ram_budget_start,
        ram_budget_end,
        ram_budget_start / base_memory,
        ram_budget_end / base_memory,
    )

    ram_budget_range = np.linspace(ram_budget_start, ram_budget_end, ram_budget_samples)

    g, *_ = make_dfgraph_costs(net=net, device=chipset)
    total_runtime = sum(g.cost_cpu.values())
    total_ram = sum(g.cost_ram[i] for i in g.vfwd)
    print(f"Total runtime of graph (forward + backward) = {total_runtime}")
    print(f"Total RAM consumption of forward pass = {total_ram}")
    print(f"### --- ### Total RAM consumption of forward pass = {total_ram}")
    print(total_threads // solve_threads)

    with ProcessPoolExecutor(max_workers=total_threads // solve_threads) as executor:
        for result in executor.map(
            solve_wrapper,
            [
                dict(
                    model=model,
                    platform=platform,
                    ram_budget=ram_budget,
                    runtime_budget=runtime_budget,
                    mem_power_scale=mem_power_scale,
                    batch_size=batch_size,
                    solver=solver,
                    time_limit_s=time_limit_s,
                    solve_threads=solve_threads,
                )
                for ram_budget in ram_budget_range
            ],
        ):
            print(
                result.total_power_cost_cpu,
                result.total_power_cost_page,
                result.ram_budget,
            )
            plt.plot(
                result.ram_budget,
                -1 if result.total_power_cost_cpu is None else result.total_power_cost_cpu + result.total_power_cost_page,
                "r.",
            )
            plt.draw()
            plt.pause(0.1)

    print("Done!")
    plt.savefig(filename)
    plt.show(block=True)


if __name__ == "__main__":
    pareto(model="resnet18_cifar", platform="m4", runtime_budget=1.2, time_limit_s=600)
