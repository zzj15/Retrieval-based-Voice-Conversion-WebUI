import logging
import os
import signal
import subprocess
import time


logger = logging.getLogger(__name__)


def kill_process_tree(process, process_name="", task_logger=None):
    """Terminate a Popen process and every child process it created."""
    if process is None:
        return False
    try:
        if process.poll() is not None:
            return False
    except (OSError, ProcessLookupError):
        return False

    pid = process.pid
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["taskkill", "/t", "/f", "/pid", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode != 0:
                process.terminate()
        except OSError:
            try:
                process.terminate()
            except (OSError, ProcessLookupError):
                pass
    else:
        try:
            process_group = os.getpgid(pid)
        except (OSError, ProcessLookupError):
            process_group = None
        try:
            if process_group is not None and process_group != os.getpgrp():
                os.killpg(process_group, signal.SIGTERM)
            else:
                os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass

        own_group = process_group is not None and process_group != os.getpgrp()
        for _ in range(10):
            running = process.poll() is None
            if own_group:
                try:
                    os.killpg(process_group, 0)
                    running = True
                except ProcessLookupError:
                    running = False
            if not running:
                break
            time.sleep(0.1)
        if own_group or process.poll() is None:
            try:
                if own_group:
                    os.killpg(process_group, signal.SIGKILL)
                else:
                    os.kill(pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass

    try:
        process.wait(timeout=5)
    except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
        try:
            process.kill()
            process.wait(timeout=5)
        except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
            pass

    log = task_logger or logger
    stopped = process.poll() is not None
    log.info("%s process tree termination %s (pid=%s)",
             process_name or "Child", "completed" if stopped else "failed", pid)
    return stopped
