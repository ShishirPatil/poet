from argparse import ArgumentParser
from poet import solve

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
    parser.add_argument("--solver", type=str, default="gurobi", choices=["gurobi", "cbc"])
    parser.add_argument("--use-actual-gurobi", action="store_true", default=False)
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
        use_actual_gurobi=args.use_actual_gurobi,
    )

    print("POET successfully found a solution!")
    print("==> R (recomputation matrix):", result["solution"].R)
    print("==> Min (Page-in matrix):", result["solution"].Min)
    print("==> Mout (Page-out matrix)", result["solution"].Mout)
