# FLOPS_PER_WATT is FLOP_PER_JOULE.
MKR1000 = {
    # Theoretical FLOPS_PER_WATT = 48 X 10^(6) / (100 X 3 X 20 X 10^(-3) X 3.3)  = 2.42 M FLOP / J
    #     Where 100 cycles for each instruction (since software simulated  -  no FPU),
    #     3 cycles per multiplication, 20 mA @ 3.3V
    "FLOPS_PER_WATT": 2000 / (0.012 * 0.020 * 3.3),  # also flop per joule
    "FLOPS": 2000 / 0.012,  # 0.16 MFLOPS
    "PAGEIN_LATENCY": 0.109 * (10 ** (-3)),  # in seconds
    "PAGEIN_THROUGHPUT": 4616.51 * (10 ** (3)),  # in bytes per second
    "PAGEOUT_LATENCY": 0.113 * (10 ** (-3)),  # in seconds
    "PAGEOUT_THROUGHPUT": 4440 * (10 ** (3)),  # in bytes per second
    "MEMORY_POWER": 100 * (10 ** (-3)) * 3.3,  # in ampere*volt
    "TYPE": 4,  # in bytes; Float - 4, Int - 4, long - 8
    "POWER": 20 * (10 ** (-3)) * 3.3,
}


M4F = {
    # Theoretical FLOPS_PER_WATT =  64 X 10^(6) / (3.328 X 10^(-3) X 3.3)
    #     Clocked at 64MHz, 3.328 mA @ 3.3V. (Assuming mult is 1 cycle)
    #     The SD card costs are same as MKR1000
    "FLOPS_PER_WATT": (16283 * 100 * 1000) / (255.6237 * 0.003328 * 3.3),  # also flop per joule
    "FLOPS": (16283 * 100 * 1000) / (255.6237),  # 6.37 MFLOPS
    "PAGEIN_LATENCY": 0.109 * (10 ** (-3)),  # in seconds
    "PAGEIN_THROUGHPUT": 4616.51 * (10 ** (3)),  # in bytes per second
    "PAGEOUT_LATENCY": 0.113 * (10 ** (-3)),  # in seconds
    "PAGEOUT_THROUGHPUT": 4440 * (10 ** (3)),  # in bytes per second
    "MEMORY_POWER": 100 * (10 ** (-3)) * 3.3,  # in ampere*volt
    "TYPE": 4,  # in bytes; Float - 4, Int - 4, long - 8
    "POWER": 3.328 * (10 ** (-3)) * 3.3,
}


RPi = {
    # Theoretical FLOPS_PER_WATT = 600 X 10^(6) X 4 X 1 / (1.25 * 5) = 384 M FLOP / J
    #     Clocked at 600MHz, with 4 cores, Single flop per cycle.
    #     Caution: 1.25 is the total power consumption, here we assume (peak-idle)
    "FLOPS_PER_WATT": (16283 * 100 * 1000) / (13.53 * 0.6 * 5),  # also flop per joule
    "FLOPS": (2 * 1000 * 1000 * 1000),  # 120.34 MFLOPS
    "PAGEIN_LATENCY": 0.00509561 * (10 ** (-3)),  # in seconds
    "PAGEIN_THROUGHPUT": 45.5 * (10 ** (6)),  # in bytes per second
    "PAGEOUT_LATENCY": 0.0185314 * (10 ** (-3)),  # in seconds (config: with caching)
    "PAGEOUT_THROUGHPUT": 24.5 * (10 ** (6)),  # in bytes per second (config: with caching)
    "MEMORY_POWER": 30 * (10 ** (-3)) * 5,  # in ampere*volt
    "TYPE": 4,  # in bytes; Float - 4, Int - 4, long - 8
    "POWER": 0.6 * 5,  # 600mA @ 5V
}


RPiNoCache = {
    # Theoretical FLOPS_PER_WATT = 600 X 10^(6) X 4 X 1 / (1.25 * 5) = 384 M FLOP / J
    #     Clocked at 600MHz, with 4 cores, Single flop per cycle.
    #     Caution: 1.25 is the total power consumption, here we assume (peak-idle)
    "FLOPS_PER_WATT": (16283 * 100 * 1000) / (13.53 * 0.6 * 5),  # also flop per joule
    "FLOPS": (16283 * 100 * 1000) / (13.53),  # 120.34 MFLOPS
    "PAGEIN_LATENCY": 0.00509561 * (10 ** (-3)),  # in seconds
    "PAGEIN_THROUGHPUT": 45.5 * (10 ** (6)),  # in bytes per second
    "PAGEOUT_LATENCY": 4.23532 * (10 ** (-3)),  # in seconds (config: without caching)
    "PAGEOUT_THROUGHPUT": 17.1 * (10 ** (6)),  # in bytes per second (config: without caching)
    "MEMORY_POWER": 30 * (10 ** (-3)) * 5,  # in ampere*volt
    "TYPE": 4,  # in bytes; Float - 4, Int - 4, long - 8
    "POWER": 0.6 * 5,  # 600mA @ 5V
}

JetsonTX2 = {
    "PAGEIN_LATENCY": 0.00054,  # in seconds
    "PAGEOUT_LATENCY": 0.002396,  # in seconds
    "POWER": (995 - 229) * (10 ** (-3)),  # stress 4cpu is 2753, 1 cpu is 995 already in Watt
    "PAGEIN_THROUGHPUT": 108.5 * (10 ** (6)),  # in bytes per second
    "PAGEOUT_THROUGHPUT": 94.37 * (10 ** (6)),  # in bytes per second
    "MEMORY_POWER": 309 * (10 ** (-3)),  # This is the read power. Considered same for simplicity
    "TYPE": 4,  # in bytes; Float - 4, Int - 4, long - 8
}
