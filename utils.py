import os
import shutil
import logging
import asyncio
import platform
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

import py3ddose

logger = logging.getLogger(__name__)

if platform.system() == 'Darwin':
    MAX = cpu_count() - 1
else:
    MAX = cpu_count()

counter = asyncio.Semaphore(MAX)

executor = ProcessPoolExecutor()


def remove(path):
    try:
        os.remove(path)
    except IOError:
        pass


async def copy(src, dst):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, shutil.copy, src, dst)


async def read_3ddose(path):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, py3ddose.read_3ddose, path)


async def run_command(command, stdin=None, **kwargs):
    await counter.acquire()
    logger.info('Running "{}"'.format(' '.join(command)))
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        stdin=asyncio.subprocess.PIPE if stdin else None,
        **kwargs)
    stdout, stderr = await process.communicate(stdin)
    stdout = stdout.decode('utf-8')
    if process.returncode != 0 or 'ERROR' in stdout or 'Warning' in stdout:
        message = 'Command failed: "{}"'.format(' '.join(command))
        logger.error(message)
        logger.error(stdout)
        raise RuntimeError(message)
    counter.release()
    return stdout


async def main():
    await run_command(['BEAM_CLMT10', '-p', 'allkV', '-i', '7e14c3496a36b37ed9fe369f7222136b.egsinp'], cwd='/Users/henry/projects/EGSnrc/egs_home/BEAM_CLMT10')

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
