import logging
import asyncio
import platform
from multiprocessing import cpu_count

logger = logging.getLogger(__name__)

if platform.system() == 'Darwin':
    MAX = cpu_count() - 1
else:
    MAX = cpu_count()

counter = asyncio.Semaphore(MAX)

async def run_command(command, **kwargs):
    await counter.acquire()
    logger.info('Running "{}"'.format(' '.join(command)))
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        **kwargs)
    await process.wait()
    out = (await process.stdout.read()).decode('utf-8')
    if process.returncode != 0 or 'ERROR' in out:
        message = 'Command failed: "{}"'.format(' '.join(command))
        logger.error(message)
        logger.error(out)
        raise RuntimeError(message)
    counter.release()
    return out

