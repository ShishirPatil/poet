import os
from contextlib import redirect_stdout
from typing import Optional

import numpy as np
from gurobipy import GRB, Model, quicksum
from loguru import logger

from poet.poet_solver import POETSolution
from poet.utils.checkmate.core.dfgraph import DFGraph
from poet.utils.checkmate.core.utils.definitions import PathLike
from poet.utils.checkmate.core.utils.timer import Timer

# noinspection PyPackageRequirements

# POET ILP defined using Gurobi
class POETSolverGurobi:
    def __init__(
        self,
        g: DFGraph,
        # cost weighting
        cpu_power_cost_vec_joule: np.ndarray,
        pagein_power_cost_vec_joule: np.ndarray,
        pageout_power_cost_vec_joule: np.ndarray,
        # constraints
        ram_budget_bytes: Optional[float] = None,
        runtime_budget_ms: Optional[float] = None,
        paging=True,
        remat=True,
        # solver options
        time_limit_s: float = 1e100,
        solve_threads: int = None,
    ):
        self.g = g
        self.T = self.g.size
        self.runtime_budget_ms = runtime_budget_ms
        self.ram_budget_bytes = ram_budget_bytes
        self.remat = remat
        self.paging = paging
        assert cpu_power_cost_vec_joule.shape == (self.T, 1)
        assert pagein_power_cost_vec_joule.shape == (self.T, 1)
        assert pageout_power_cost_vec_joule.shape == (self.T, 1)

        self.timer = Timer

        self.m = Model(self.model_name)
        self.m.Params.TimeLimit = time_limit_s
        if solve_threads:
            self.m.Params.Threads = solve_threads

        grb_ram_ub = ram_budget_bytes if ram_budget_bytes is not None else GRB.INFINITY

        self.R = self.m.addVars(self.T, self.T, name="R", vtype=GRB.BINARY)
        self.SRam = self.m.addVars(self.T, self.T, name="SRam", vtype=GRB.BINARY)
        self.SSd = self.m.addVars(self.T, self.T, name="SSd", vtype=GRB.BINARY)
        self.MIn = self.m.addVars(self.T, self.T, name="MIn", vtype=GRB.BINARY)
        self.MOut = self.m.addVars(self.T, self.T, name="MOut", vtype=GRB.BINARY)
        self.U = self.m.addVars(self.T, self.T, name="U", vtype=GRB.CONTINUOUS, lb=0, ub=grb_ram_ub)
        self.Free_E = self.m.addVars(self.T, len(self.g.edge_list), name="Free_E", vtype=GRB.BINARY)

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
        total_cpu_power = quicksum(self.R[t, i] * cpu_cost[i] for t in range(self.T) for i in range(self.T))
        total_pagein_power = quicksum(self.MIn[t, i] * pagein_cost[i] for t in range(self.T) for i in range(self.T))
        total_pageout_power = quicksum(self.MOut[t, i] * pageout_cost[i] for t in range(self.T) for i in range(self.T))
        self.m.setObjective(total_cpu_power + total_pagein_power + total_pageout_power, GRB.MINIMIZE)

    def _initialize_variables(self):
        self.m.addLConstr(
            quicksum(self.R[t, i] for t in range(self.T) for i in range(t + 1, self.T)),
            GRB.EQUAL,
            0,
        )
        self.m.addLConstr(
            quicksum(self.SRam[t, i] for t in range(self.T) for i in range(t, self.T)),
            GRB.EQUAL,
            0,
        )
        self.m.addLConstr(
            quicksum(self.SSd[t, i] for t in range(self.T) for i in range(t, self.T)),
            GRB.EQUAL,
            0,
        )
        self.m.addLConstr(
            quicksum(self.MIn[t, i] for t in range(self.T) for i in range(t + 1, self.T)),
            GRB.EQUAL,
            0,
        )
        self.m.addLConstr(
            quicksum(self.MOut[t, i] for t in range(self.T) for i in range(t + 1, self.T)),
            GRB.EQUAL,
            0,
        )
        self.m.addLConstr(quicksum(self.R[t, t] for t in range(self.T)), GRB.EQUAL, self.T)

    def _disable_remat(self):
        for t in range(self.T):
            for i in range(self.T):
                self.m.addLConstr(self.R[t, i], GRB.EQUAL, True if t == i else False)

    def _disable_paging(self):
        for t in range(self.T):
            for i in range(self.T):
                self.m.addLConstr(self.MIn[t, i], GRB.EQUAL, False)
                self.m.addLConstr(self.MOut[t, i], GRB.EQUAL, False)
                self.m.addLConstr(self.SSd[t, i], GRB.EQUAL, False)

    def _create_correctness_constraints(self):
        # ensure all computations are possible
        for (u, v) in self.g.edge_list:
            for t in range(self.T):
                self.m.addLConstr(self.R[t, v], GRB.LESS_EQUAL, self.R[t, u] + self.SRam[t, u])
        # ensure all checkpoints are in memory
        for t in range(self.T - 1):
            for i in range(self.T):
                self.m.addLConstr(
                    self.SRam[t + 1, i],
                    GRB.LESS_EQUAL,
                    self.SRam[t, i] + self.R[t, i] + self.MIn[t, i],
                )
                self.m.addLConstr(
                    self.SSd[t + 1, i],
                    GRB.LESS_EQUAL,
                    self.SSd[t, i] + self.MOut[t, i],
                )
        for i in range(self.T):
            for j in range(self.T):
                self.m.addLConstr(self.MIn[i, j], GRB.LESS_EQUAL, self.SSd[i, j])
                self.m.addLConstr(self.MOut[i, j], GRB.LESS_EQUAL, self.SRam[i, j])

    def _create_ram_memory_constraints(self, ram_vec_bytes):
        def _num_hazards(t, i, k):
            if t + 1 < self.T:
                return 1 - self.R[t, k] + self.SRam[t + 1, i] + quicksum(self.R[t, j] for j in self.g.successors(i) if j > k)
            return 1 - self.R[t, k] + quicksum(self.R[t, j] for j in self.g.successors(i) if j > k)

        def _max_num_hazards(t, i, k):
            num_uses_after_k = sum(1 for j in self.g.successors(i) if j > k)
            if t + 1 < self.T:
                return 2 + num_uses_after_k
            return 1 + num_uses_after_k

        for t in range(self.T):
            for eidx, (i, k) in enumerate(self.g.edge_list):
                self.m.addLConstr(1 - self.Free_E[t, eidx], GRB.LESS_EQUAL, _num_hazards(t, i, k))
        for t in range(self.T):
            for eidx, (i, k) in enumerate(self.g.edge_list):
                self.m.addLConstr(
                    _max_num_hazards(t, i, k) * (1 - self.Free_E[t, eidx]),
                    GRB.GREATER_EQUAL,
                    _num_hazards(t, i, k),
                )
        for t in range(self.T):
            self.m.addLConstr(
                self.U[t, 0],
                GRB.EQUAL,
                self.R[t, 0] * ram_vec_bytes[0] + quicksum(self.SRam[t, i] * ram_vec_bytes[i] for i in range(self.T)),
            )
        for t in range(self.T):
            for k in range(self.T - 1):
                mem_freed = quicksum(ram_vec_bytes[i] * self.Free_E[t, eidx] for (eidx, i) in self.g.predecessors_indexed(k))
                self.m.addLConstr(
                    self.U[t, k + 1],
                    GRB.EQUAL,
                    self.U[t, k] + self.R[t, k + 1] * ram_vec_bytes[k + 1] - mem_freed,
                )

    def _create_runtime_constraints(self, runtime_cost_vec, runtime_budget_ms):
        total_runtime = quicksum(self.R[t, i] * runtime_cost_vec[i] for t in range(self.T) for i in range(self.T))
        self.m.addLConstr(total_runtime, GRB.LESS_EQUAL, runtime_budget_ms)

    @property
    def model_name(self):
        return (
            f"POETSolverGurobi(T={self.T}, "
            f"ram_budget_bytes={self.ram_budget_bytes}, "
            f"runtime_budget_bytes={self.runtime_budget_ms}, "
            f"paging={self.paging}, "
            f"remat={self.remat})"
        )

    def save_model(self, out_file: PathLike):
        self.m.write(str(out_file))

    def get_result(self, grb_var, dims, dtype=np.int32):
        if self.m.status == GRB.INFEASIBLE or self.m.solCount < 1:
            return None
        assert len(dims) == 2
        rows, cols = dims
        out_data = np.zeros(dims, dtype=dtype)
        for i in range(rows):
            for j in range(cols):
                try:
                    out_data[i, j] = grb_var[i, j].X
                except (AttributeError, TypeError) as e:
                    logger.error(f"Got exception when parsing Gurobi model outputs, {e}")
                    out_data[i, j] = grb_var[i, j]
        return out_data.tolist()

    def solve(
        self,
        write_file: Optional[PathLike] = None,
        extra_gurobi_params={},
    ):
        with self.timer("Optimizer flags"):
            if write_file is not None:
                self.m.Params.LogToConsole = 0
                self.m.Params.LogFile = str(write_file)
            else:
                self.m.Params.OutputFlag = 0
            for k, v in extra_gurobi_params:
                setattr(self.m.Params, k, v)
        with self.timer("ILP solve"):
            with Timer("solve_timer") as t:
                with redirect_stdout(open(os.devnull, "w")):
                    self.m.optimize()
        solve_time = t.elapsed

        is_feasible = self.m.status != GRB.INFEASIBLE and self.m.solCount >= 1
        return POETSolution(
            R=self.get_result(self.R, (self.T, self.T)),
            SRam=self.get_result(self.SRam, (self.T, self.T)),
            SSd=self.get_result(self.SSd, (self.T, self.T)),
            Min=self.get_result(self.MIn, (self.T, self.T)),
            Mout=self.get_result(self.MOut, (self.T, self.T)),
            FreeE=self.get_result(self.Free_E, (self.T, len(self.g.edge_list))),
            U=self.get_result(self.U, (self.T, self.T), dtype=float),
            finished=self.m.status in [GRB.OPTIMAL, GRB.INFEASIBLE],
            feasible=is_feasible,
            solve_time_s=solve_time,
        )
