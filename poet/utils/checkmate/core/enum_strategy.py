from enum import Enum


class SolveStrategy(Enum):
    NOT_SPECIFIED = "NOT_SPECIFIED"
    CHEN_SQRTN = "CHEN_SQRTN"
    CHEN_GREEDY = "CHEN_GREEDY"
    CHEN_SQRTN_NOAP = "CHEN_SQRTN_NOAP"
    CHEN_GREEDY_NOAP = "CHEN_GREEDY_NOAP"
    OPTIMAL_ILP_GC = "OPTIMAL_ILP_GC"
    CHECKPOINT_LAST_NODE = "CHECKPOINT_LAST_NODE"
    CHECKPOINT_ALL = "CHECKPOINT_ALL"
    CHECKPOINT_ALL_AP = "CHECKPOINT_ALL_AP"
    GRIEWANK_LOGN = "GRIEWANK_LOGN"
    APPROX_DET_ROUND_LP_SWEEP = "APPROX_DET_ROUND_LP_SWEEP"
    APPROX_DET_ROUND_LP_05_THRESH = "APPROX_DET_ROUND_LP_05_THRESH"
    APPROX_DET_RANDOM_THRESH_ROUND_LP = "APPROX_DET_RANDOM_THRESH_ROUND_LP"
    APPROX_RANDOMIZED_ROUND = "APPROX_RANDOMIZED_ROUND"
    LB_LP = "LB_LP"
    SIMRD = "SIMRD"
    SIMRD_MSPS = "SIMRD_MSPS"

    @classmethod
    def get_description(cls, val, model_name=None):
        is_linear = model_name in ("VGG16", "VGG19", "MobileNet")
        return {
            cls.CHEN_SQRTN: "AP $\\sqrt{n}$",
            cls.CHEN_GREEDY: "AP greedy",
            cls.CHEN_SQRTN_NOAP: "Generalized $\\sqrt{n}$" if not is_linear else "Chen et al. $\\sqrt{n}$",
            cls.CHEN_GREEDY_NOAP: "Generalized greedy",
            cls.OPTIMAL_ILP_GC: "Optimal MILP (proposed)",
            cls.CHECKPOINT_LAST_NODE: "Checkpoint last node",
            cls.CHECKPOINT_ALL: "Checkpoint all (ideal)",
            cls.CHECKPOINT_ALL_AP: "Checkpoint all APs",
            cls.GRIEWANK_LOGN: "Griewank et al. $\\log~n$" if is_linear else "AP $\\log~n$",
            cls.APPROX_DET_ROUND_LP_SWEEP: "Approximation via deterministic rounding of LP relaxation w/ threshold sweep",
            cls.APPROX_DET_RANDOM_THRESH_ROUND_LP: "Approximation via deterministic rounding of LP relaxation with random thresholds",
            cls.APPROX_DET_ROUND_LP_05_THRESH: "Approximation via deterministic rounding of LP relaxation w/ 0.5 threshold",
            cls.APPROX_RANDOMIZED_ROUND: "Approximation via randomized rounding of LP relaxation",
            cls.LB_LP: "Lower bound via LP relaxation",
            cls.SIMRD: "Dynamic Tensor Rematerialization",
            cls.SIMRD_MSPS: "Capuchin MSPS heuristic from DTR",
        }[val]

    # todo move this to experiments codebase
    @classmethod
    def get_plot_params(cls, val):
        from matplotlib import rcParams

        fullsize = rcParams["lines.markersize"]
        halfsize = fullsize / 2
        bigger = fullsize * 1.5
        mapping = {
            cls.CHEN_SQRTN: ("c", "D", halfsize),
            cls.CHEN_SQRTN_NOAP: ("c", "^", halfsize),
            cls.CHEN_GREEDY: ("g", ".", fullsize),
            cls.CHEN_GREEDY_NOAP: ("g", "+", fullsize),
            cls.CHECKPOINT_ALL: ("k", "*", bigger),
            cls.CHECKPOINT_ALL_AP: ("b", "x", fullsize),
            cls.GRIEWANK_LOGN: ("m", "p", fullsize),
            cls.OPTIMAL_ILP_GC: ("r", "s", halfsize),
            cls.APPROX_DET_ROUND_LP_SWEEP: ("r", "*", fullsize),
            cls.APPROX_DET_ROUND_LP_05_THRESH: ("r", "^", halfsize),
            cls.APPROX_DET_RANDOM_THRESH_ROUND_LP: ("r", "x", fullsize),
            cls.APPROX_RANDOMIZED_ROUND: ("r", "+", fullsize),
            cls.LB_LP: ("r", "p", fullsize),
            cls.SIMRD: ("r", ".", fullsize),
            cls.SIMRD_MSPS: ("m", ".", fullsize),
        }
        if val in mapping:
            return mapping[val]
        raise NotImplementedError("No plotting parameters for strategy {}".format(val))


class ImposedSchedule(Enum):
    COVER_LAST_NODE = "COVER_LAST_NODE"
    COVER_ALL_NODES = "COVER_ALL_NODES"
    FULL_SCHEDULE = "FULL_SCHEDULE"

    def __str__(self):
        return self.value
