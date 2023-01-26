import os
from typing import Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from loguru import logger

from poet import solve

SOLVE_THREADS = min(4, os.cpu_count())

app = FastAPI()


@app.get("/solve")
def solve_handler(
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
    time_limit_s: float = 1e100,
    solve_threads: int = SOLVE_THREADS,  # different default than a direct solve
):
    try:
        return solve(
            model=model,
            platform=platform,
            ram_budget=ram_budget,
            runtime_budget=runtime_budget,
            paging=paging,
            remat=remat,
            mem_power_scale=mem_power_scale,
            batch_size=batch_size,
            solver=solver,
            time_limit_s=time_limit_s,
            solve_threads=solve_threads,
            print_graph_info=False,
        )
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logger.info("Initializing an instance of the POET server.")
    uvicorn.run("server:app", host="0.0.0.0", port=80, reload=os.environ.get("DEV"))
