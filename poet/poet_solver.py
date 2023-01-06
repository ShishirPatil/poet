from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pulp as pl

from poet.utils.checkmate.core.dfgraph import DFGraph
from poet.utils.checkmate.core.utils.timer import Timer


# POET ILP defined using PuLP
@dataclass
class POETSolution:
    R: np.ndarray
    SRam: np.ndarray
    SSd: np.ndarray
    Min: np.ndarray
    Mout: np.ndarray
    FreeE: np.ndarray
    U: np.ndarray
    optimal: bool
    feasible: bool
    solve_time_s: Optional[float] = float("inf")


class POETSolver:
    def __init__(
        self,
        g: DFGraph,
        # cost weighting
        cpu_power_cost_vec_joule,
        pagein_power_cost_vec_joule,
        pageout_power_cost_vec_joule,
        # constraints
        ram_budget_bytes: Optional[float] = None,
        runtime_budget_ms: Optional[float] = None,
        paging=True,
        remat=True,
        solver: Optional[Literal["cbc", "gurobi"]] = None,
        time_limit_s: Optional[float] = None,
        solve_threads: Optional[int] = None,
    ):
        self.g = g
        self.m = pl.LpProblem("POET", pl.LpMinimize)
        self.T = g.size
        gurobi_available = "GUROBI" in pl.listSolvers(onlyAvailable=True)
        self.solver = pl.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s, threads=solve_threads)
        if solver == "gurobi":
            if gurobi_available:
                self.solver = pl.GUROBI(msg=False, timeLimit=time_limit_s, solve_threads=solve_threads)
            else:
                print("Warning: Gurobi not available or has an incorrect installation; falling back to CBC solver")

        self.runtime_budget_ms = runtime_budget_ms
        self.ram_budget_bytes = ram_budget_bytes
        self.remat = remat
        self.paging = paging

        assert cpu_power_cost_vec_joule.shape == (self.T, 1)
        assert pagein_power_cost_vec_joule.shape == (self.T, 1)
        assert pageout_power_cost_vec_joule.shape == (self.T, 1)

        grb_ram_ub = ram_budget_bytes if ram_budget_bytes is not None else float("inf")

        self.R = [[pl.LpVariable(f"R_{i},{j}", None, None, pl.LpBinary) for j in range(self.T)] for i in range(self.T)]
        self.SRam = [[pl.LpVariable(f"SRam_{i},{j}", None, None, pl.LpBinary) for j in range(self.T)] for i in range(self.T)]
        self.SSd = [[pl.LpVariable(f"SSd_{i},{j}", None, None, pl.LpBinary) for j in range(self.T)] for i in range(self.T)]
        self.MIn = [[pl.LpVariable(f"MIn_{i},{j}", None, None, pl.LpBinary) for j in range(self.T)] for i in range(self.T)]
        self.MOut = [[pl.LpVariable(f"MOut_{i},{j}", None, None, pl.LpBinary) for j in range(self.T)] for i in range(self.T)]
        self.U = [[pl.LpVariable(f"U_{i},{j}", 0, grb_ram_ub, pl.LpContinuous) for j in range(self.T)] for i in range(self.T)]
        self.Free_E = [[pl.LpVariable(f"Free_E_{i},{j}", None, None, pl.LpBinary) for j in range(len(g.edge_list))] for i in range(self.T)]

        self._create_objective(
            cpu_power_cost_vec_joule,
            pagein_power_cost_vec_joule,
            pageout_power_cost_vec_joule,
        )
        self._initialize_variables()
        self._create_correctness_constraints()
        if ram_budget_bytes is not None:
            ram_vec_bytes = [self.g.cost_ram[i] for i in range(self.T)]
            self._create_ram_memory_constraints(ram_vec_bytes)
        if runtime_budget_ms is not None:
            runtime_vec_ms = [self.g.cost_cpu[i] for i in range(self.T)]
            self._create_runtime_constraints(runtime_vec_ms, runtime_budget_ms)
        if not self.paging:
            self._disable_paging()
        if not self.remat:
            self._disable_remat()

    def _create_objective(self, cpu_cost, pagein_cost, pageout_cost):
        total_cpu_power = pl.lpSum(self.R[t][i] * cpu_cost[i] for t in range(self.T) for i in range(self.T))
        total_pagein_power = pl.lpSum(self.MIn[t][i] * pagein_cost[i] for t in range(self.T) for i in range(self.T))
        total_pageout_power = pl.lpSum(self.MOut[t][i] * pageout_cost[i] for t in range(self.T) for i in range(self.T))
        self.m += total_cpu_power + total_pagein_power + total_pageout_power

    def _initialize_variables(self):
        for constraint in [
            pl.lpSum(self.R[t][i] for t in range(self.T) for i in range(t + 1, self.T)) == 0,
            pl.lpSum(self.SRam[t][i] for t in range(self.T) for i in range(t, self.T)) == 0,
            pl.lpSum(self.SSd[t][i] for t in range(self.T) for i in range(t, self.T)) == 0,
            pl.lpSum(self.MIn[t][i] for t in range(self.T) for i in range(t + 1, self.T)) == 0,
            pl.lpSum(self.MOut[t][i] for t in range(self.T) for i in range(t + 1, self.T)) == 0,
            pl.lpSum(self.R[t][t] for t in range(self.T)) == self.T,
        ]:
            self.m += constraint

    def _create_correctness_constraints(self):
        # ensure all computations are possible
        for (u, v) in self.g.edge_list:
            for t in range(self.T):
                self.m += self.R[t][v] <= self.R[t][u] + self.SRam[t][u]
        # ensure all checkpoints are in memory
        for t in range(self.T - 1):
            for i in range(self.T):
                self.m += self.SRam[t + 1][i] <= self.SRam[t][i] + self.R[t][i] + self.MIn[t][i]
                self.m += self.SSd[t + 1][i] <= self.SSd[t][i] + self.MOut[t][i]

        for i in range(self.T):
            for j in range(self.T):
                self.m += self.MIn[i][j] <= self.SSd[i][j]
                self.m += self.MOut[i][j] <= self.SRam[i][j]

    def _create_ram_memory_constraints(self, ram_vec_bytes):
        def _num_hazards(t, i, k):
            if t + 1 < self.T:
                return 1 - self.R[t][k] + self.SRam[t + 1][i] + pl.lpSum(self.R[t][j] for j in self.g.successors(i) if j > k)
            return 1 - self.R[t][k] + pl.lpSum(self.R[t][j] for j in self.g.successors(i) if j > k)

        def _max_num_hazards(t, i, k):
            num_uses_after_k = sum(1 for j in self.g.successors(i) if j > k)
            if t + 1 < self.T:
                return 2 + num_uses_after_k
            return 1 + num_uses_after_k

        # upper and lower bounds for 1 - Free_E
        for t in range(self.T):
            for eidx, (i, k) in enumerate(self.g.edge_list):
                self.m += 1 - self.Free_E[t][eidx] <= _num_hazards(t, i, k)
                self.m += _max_num_hazards(t, i, k) * (1 - self.Free_E[t][eidx]) >= _num_hazards(t, i, k)
        # initialize memory usage (includes spurious checkpoints)
        for t in range(self.T):
            self.m += self.U[t][0] == self.R[t][0] * ram_vec_bytes[0] + pl.lpSum(self.SRam[t][i] * ram_vec_bytes[i] for i in range(self.T))

        # memory recurrence
        for t in range(self.T):
            for k in range(self.T - 1):
                mem_freed = pl.lpSum(ram_vec_bytes[i] * self.Free_E[t][eidx] for (eidx, i) in self.g.predecessors_indexed(k))
                self.m += self.U[t][k + 1] == self.U[t][k] + self.R[t][k + 1] * ram_vec_bytes[k + 1] - mem_freed

    def _create_runtime_constraints(self, runtime_cost_vec, runtime_budget_ms):
        total_runtime = pl.lpSum(self.R[t][i] * runtime_cost_vec[i] for t in range(self.T) for i in range(self.T))
        self.m += total_runtime <= runtime_budget_ms

    def _disable_paging(self):
        for t in range(self.T):
            for i in range(self.T):
                self.m += self.MIn[t][i] == False
                self.m += self.MOut[t][i] == False
                self.m += self.SSd[t][i] == False

    def _disable_remat(self):
        for t in range(self.T):
            for i in range(self.T):
                self.m += self.R[t][i] == True if t == i else False

    def is_feasible(self):
        return self.m.status not in [
            pl.LpStatusInfeasible,
            pl.LpStatusNotSolved,
            pl.LpStatusUndefined,
            pl.LpStatusUnbounded,
        ]

    def get_result(self, var_matrix, dtype=int):
        if not self.is_feasible():
            return None
        return [[dtype(pl.value(var_matrix[i][j])) for j in range(len(var_matrix[0]))] for i in range(len(var_matrix))]

    def solve(self):
        with Timer("solve_timer") as t:
            self.m.solve(self.solver)
        return POETSolution(
            R=self.get_result(self.R),
            SRam=self.get_result(self.SRam),
            SSd=self.get_result(self.SSd),
            Min=self.get_result(self.MIn),
            Mout=self.get_result(self.MOut),
            FreeE=self.get_result(self.Free_E),
            U=self.get_result(self.U, dtype=float),
            optimal=self.m.status == pl.LpStatusOptimal,
            feasible=self.is_feasible(),
            solve_time_s=t.elapsed,
        )

    @property
    def model_name(self):
        return (
            f"POETSolver(T={self.T}, "
            f"ram_budget_bytes={self.ram_budget_bytes}, "
            f"runtime_budget_bytes={self.runtime_budget_ms}, "
            f"paging={self.paging}, "
            f"remat={self.remat})"
        )
