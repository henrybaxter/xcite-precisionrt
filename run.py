from simulator.utils import run_async
from simulator.run import main

if __name__ == '__main__':
    run_async(main())
    # import warnings
    # loop.set_debug(True)
    # loop.slow_callback_duration = 0.001
    # warnings.simplefilter('always', ResourceWarning)
