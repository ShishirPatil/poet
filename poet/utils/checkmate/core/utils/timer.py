from decimal import Decimal
from math import ceil, log10
from timeit import default_timer


class Timer:
    """
    A context manager class for measuring and printing elapsed time durations.

    Attributes:
        niters (int): Number of iterations.
        _elapsed (Decimal): Total elapsed time.
        _name (str): Name of the timer.
        _print_results (bool): Whether to print results after timing.
        _start_time (float): Start time of the timer.
        _children (dict): Dictionary to hold child timers.
        _count (int): Number of times the timer has been started.
    """
    def __init__(self, name, extra_data=None, print_results=False, niters=None):
        """
        Initialize a Timer object.
        """
        self.niters = niters
        self._elapsed = Decimal()
        self._name = name
        if extra_data:
            self._name += "; " + str(extra_data)
        self._print_results = print_results
        self._start_time = None
        self._children = {}
        self._count = 0

    @property
    def elapsed(self):
        """Return the total elapsed time in seconds."""
        return float(self._elapsed)

    def __enter__(self):
        """Enter the context manager and start the timer."""
        self.start()
        return self

    def __exit__(self, *_):
        """Exit the context manager and stop the timer."""
        self.stop()
        if self._print_results:
            self.print_results()

    def child(self, name):
        """
        Create a child Timer with a given name.

        Returns:
            Timer: Child Timer instance.
        """
        try:
            return self._children[name]
        except KeyError:
            result = Timer(name, print_results=False)
            self._children[name] = result
            return result

    def start(self):
        self._count += 1
        self._start_time = self._get_time()

    def stop(self):
        self._elapsed += self._get_time() - self._start_time

    def print_results(self):
        print(self._format_results())

    def _format_results(self, indent="  "):
        """
        Format the timer results for printing.

        Returns:
            str: Formatted timer results.
        """
        children = self._children.values()
        elapsed = self._elapsed or sum(c._elapsed for c in children)
        result = "%s: %.3fs" % (self._name, elapsed)
        max_count = max(c._count for c in children) if children else 0
        count_digits = 0 if max_count <= 1 else int(ceil(log10(max_count + 1)))
        for child in sorted(children, key=lambda c: c._elapsed, reverse=True):
            lines = child._format_results(indent).split("\n")
            child_percent = child._elapsed / elapsed * 100
            lines[0] += " (%d%%)" % child_percent
            if count_digits:
                # `+2` for the 'x' and the space ' ' after it:
                lines[0] = ("%dx " % child._count).rjust(count_digits + 2) + lines[0]
            for line in lines:
                result += "\n" + indent + line
        return result

    @staticmethod
    def _get_time():
        return Decimal(default_timer())
