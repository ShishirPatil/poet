import re
import subprocess

def test_readme_example():
    command = "python poet/solve.py --model resnet18_cifar --platform a72 --ram-budget 3000000 --runtime-budget 7.6"
    output = subprocess.check_output(command, shell=True).decode("utf-8")
    assert re.test(r"POET successfully found an optimal solution with a memory budget of 3000000 bytes that consumes 7.8\d+ J of CPU power and 0.001\d+ J of memory paging power", output)
