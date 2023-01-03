import pickle
from typing import NamedTuple, Dict, List, Union, Optional

import numpy as np

from poet.utils.checkmate.core.enum_strategy import SolveStrategy, ImposedSchedule


class OperatorEvaluation(NamedTuple):
    id: int
    arg_regs: Dict[int, int]
    out_register: int
    operator_cost: int
    is_backwards: bool = False
    update_aux_vars: bool = True  # will be true if this is the last time this node is evaluated


class AllocateRegister(NamedTuple):
    register_id: int
    for_operation_id: int
    register_size: int


class DeallocateRegister(NamedTuple):
    op_id: int
    register_id: int


Schedule = List[Union[OperatorEvaluation, AllocateRegister, DeallocateRegister]]


class SchedulerAuxData(NamedTuple):
    R: np.ndarray
    S: np.ndarray
    cpu: int
    peak_ram: int  # includes memory for params
    activation_ram: int
    mem_grid: np.ndarray
    mem_timeline: List[int]
    schedule_time_s: Optional[float] = None


class ILPAuxData(NamedTuple):
    ilp_num_variables: int
    ilp_num_constraints: int
    ilp_approx: bool
    ilp_eps_noise: float
    U: Optional[np.ndarray] = None
    Free_E: Optional[np.ndarray] = None
    ilp_time_limit: Optional[int] = None
    ilp_imposed_schedule: Optional[ImposedSchedule] = None

    # approximation results
    approx_deterministic_round_threshold: Optional[float] = None


class ScheduledResult(NamedTuple):
    solve_strategy: SolveStrategy
    solver_budget: float
    feasible: bool

    schedule: Optional[Schedule] = None
    schedule_aux_data: Optional[SchedulerAuxData] = None
    ilp_aux_data: Optional[ILPAuxData] = None
    solve_time_s: Optional[float] = None

    def dumps(self) -> bytes:
        return pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def loads(serialized_result: bytes) -> "ScheduledResult":  # forward ref using string
        return pickle.loads(serialized_result)
