import itertools
import logging
from typing import List, Dict, Tuple, Optional

import numpy as np

from poet.utils.checkmate.core.dfgraph import DFGraph
from poet.utils.checkmate.core.schedule import OperatorEvaluation, AllocateRegister, DeallocateRegister, Schedule, SchedulerAuxData
from poet.utils.checkmate.core.utils.timer import Timer


class InfeasibleScheduleError(ValueError):
    pass


class ScheduleBuilder:
    def __init__(self, g, verbosity: int = 2):
        self.max_ram = 0
        self.current_ram = 0
        self.total_cpu = 0
        self.g = g
        self.schedule = []  # type: Schedule
        self.live_registers = {}  # type: Dict[int, int]
        self.next_free_register_id = 0
        self.verbosity = verbosity
        self.ram_timeline = []  # type: List[int]
        self.allocate_register(0)

    def is_op_cached(self, op_id: int):
        return op_id in self.live_registers.keys()

    def allocate_register(self, op_id: int):
        """
        Schedule a register allocation
        :param op_id: ID for operation whose output will be stored in this register,
        :return: the newly allocated register ID
        """
        if op_id in self.live_registers.keys():
            if self.verbosity >= 2:
                logging.error("Double alloc register for op #{}, reusing reg #{}".format(op_id, self.live_registers[op_id]))
            return self.live_registers[op_id]
        reg = AllocateRegister(self.next_free_register_id, op_id, self.g.cost_ram[op_id])
        self.live_registers[op_id] = reg.register_id
        self.schedule.append(reg)
        self.next_free_register_id += 1
        self.max_ram = max(self.max_ram, self.current_ram)
        self.ram_timeline.append(self.current_ram)
        self.current_ram += self.g.cost_ram[op_id]
        return reg.register_id

    def run_operator(self, op_id: int, update_aux_vars: bool):
        if not all([pred == 0 or pred in self.live_registers.keys() for pred in self.g.predecessors(op_id)]):
            raise InfeasibleScheduleError(
                "Dependency not fulfilled for op #{}, ops in ram now are {} but I need {}".format(
                    op_id, set(self.live_registers.keys()), self.g.predecessors(op_id)
                )
            )
        out_reg = self.allocate_register(op_id)
        in_regs = {pred_id: self.live_registers[pred_id] for pred_id in self.g.predecessors(op_id) if pred_id != 0}
        eval_op = OperatorEvaluation(
            op_id,
            in_regs,
            out_reg,
            self.g.cost_cpu[op_id],
            update_aux_vars=update_aux_vars,
            is_backwards=op_id not in self.g.vfwd,
        )
        self.schedule.append(eval_op)
        self.total_cpu += self.g.cost_cpu[op_id]
        self.ram_timeline.append(self.current_ram)

    def deallocate_register(self, op_id: int):
        """
        Schedule a register deallocation
        :param op_id: ID for operation whose output will be stored in this register
        """
        if op_id not in self.live_registers.keys():
            print("WARNING! Double free output register for op #{}".format(op_id))
        reg_id = self.live_registers.pop(op_id)
        self.schedule.append(DeallocateRegister(op_id, reg_id))
        self.current_ram -= self.g.cost_ram[op_id]
        self.ram_timeline.append(self.current_ram)


def print_2d_array(arr, true_char="#", false_char="_"):
    string = ""
    for row in arr:
        for val in row:
            string += true_char if val else false_char
        string += "\n"
    return string


def schedule_from_rs(g: DFGraph, r: np.ndarray, s: np.ndarray) -> Tuple[Optional[Schedule], Optional[SchedulerAuxData]]:
    # debug_collect_ram_usage = "DEBUG_SCHEDULER_RAM" in active_env_var_flags
    debug_collect_ram_usage = True
    if r is None or s is None:
        return None, None  # infeasible
    T = g.size

    def _used_after(t_, u_, i_):
        """Returns True if v_u is used after v_i in stage t"""
        is_retained_snapshot = t_ < T - 1 and s[t_ + 1, u_] == 1
        is_used_by_successor = not all([r[t_, v] == 0 or v <= i_ for v in g.successors(u_)])
        return is_retained_snapshot or is_used_by_successor

    with Timer("schedule_rs_matrix") as schedule_timer:
        # compute last usage to determine whether to update auxiliary variables
        # last_used = {i: max([t for t in range(T) if r[t, i] == 1]) for i in range(T)}
        mem_usage = np.zeros((T, T), dtype=np.int)
        sb = ScheduleBuilder(g, verbosity=1)
        for t in range(T):  # For each Timestep 'T'
            # Free unused checkpoints
            if debug_collect_ram_usage:
                for i in filter(lambda x: sb.is_op_cached(x), range(T)):
                    if not _used_after(t, i, i):
                        sb.deallocate_register(i)

            for i in range(T):  # For each node in graph
                if r[t, i] == 1:
                    # sb.run_operator(i, last_used[i] == t)
                    sb.run_operator(i, False)  # todo(paras) prune away last_used in favor of recompute blacklist
                if debug_collect_ram_usage:
                    mem_usage[t, i] = sb.current_ram + g.cost_ram_fixed
                    # g.cost_ram_fixed: Cost of parameters, etc
                    # sb.current_ram: activations

                # Free memory
                if debug_collect_ram_usage:
                    for u in filter(lambda x: sb.is_op_cached(x), itertools.chain(g.predecessors(i), [i])):
                        if not _used_after(t, u, i):
                            sb.deallocate_register(u)
        total_ram = sb.max_ram + g.cost_ram_fixed
        ram_timeline = [mem + g.cost_ram_fixed for mem in sb.ram_timeline]

    return (
        sb.schedule,
        SchedulerAuxData(
            R=r,
            S=s,
            cpu=sb.total_cpu,
            peak_ram=total_ram,
            activation_ram=sb.max_ram,
            mem_grid=mem_usage,
            mem_timeline=ram_timeline,
            schedule_time_s=schedule_timer.elapsed,
        ),
    )
