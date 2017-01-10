import asyncio

from simulator.run import main

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    # import warnings
    # loop.set_debug(True)
    # loop.slow_callback_duration = 0.001
    # warnings.simplefilter('always', ResourceWarning)
    loop.run_until_complete(main())
    loop.close()
