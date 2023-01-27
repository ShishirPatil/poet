import os
from multiprocessing import Manager
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.applications import JSONResponse
from loguru import logger

from poet import solve

SOLVE_THREADS = min(4, os.cpu_count())
MAX_THREADS = SOLVE_THREADS
NUM_WORKERS = 2 * os.cpu_count() + 1

# makes ANSI color codes work on Windows
os.system("")

app = FastAPI()
# TODO: this doesn't work with workers since each worker has its own threads_used
# see https://stackoverflow.com/questions/65686318/sharing-python-objects-across-multiple-workers/65699375#65699375
# need to implement Redis cache or something similar
threads_used = None


@app.get("/solve")
def solve_handler(
    request: Request,
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
    host = request.client.host
    if host not in threads_used:
        threads_used[host] = 0
    if threads_used[host] + solve_threads > MAX_THREADS:
        return JSONResponse(
            status_code=429,
            content={
                "detail": "Too many threads requested for solves by this user "
                + f"({threads_used[host]} in use; solving with {solve_threads} more would "
                + f"exceed the max per user of {MAX_THREADS} threads). Please wait until "
                + "some of your solves finish, or retry with less solve_threads."
            },
        )

    threads_used[host] += solve_threads

    try:
        result = solve(
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
        threads_used[host] -= solve_threads
        if threads_used[host] == 0:
            del threads_used[host]
        return result
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logger.info("Initializing an instance of the POET server.")
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=80,
        reload=os.environ.get("DEV"),
        workers=NUM_WORKERS,
    )
