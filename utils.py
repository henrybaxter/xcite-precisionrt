import logging
import asyncio
import platform
from multiprocessing import cpu_count

import py3ddose

logger = logging.getLogger(__name__)

if platform.system() == 'Darwin':
    MAX = cpu_count() - 1
else:
    MAX = cpu_count()

counter = asyncio.Semaphore(MAX)


async def read_3ddose(path):
    await counter.acquire()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, py3ddose.read_3ddose, path)


async def run_command(command, **kwargs):
    await counter.acquire()
    logger.info('Running "{}"'.format(' '.join(command)))
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        **kwargs)
    stdout, stderr = await process.communicate(None)
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
