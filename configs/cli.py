import argparse
import sys


def parse_launch_args(argv=None):
    """Parse WebUI/realtime launch flags only when an entrypoint requests it."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", dest="listen_port", metavar="PORT", type=int, default=7865,
        help="Listen port",
    )
    parser.add_argument(
        "--pycmd", dest="python_cmd", metavar="PYCMD", default=sys.executable or "python",
        help="Python command",
    )
    parser.add_argument(
        "--colab", dest="iscolab", action="store_true", help="Launch in colab"
    )
    parser.add_argument(
        "--noparallel", action="store_true", help="Disable parallel processing"
    )
    parser.add_argument(
        "--noautoopen", action="store_true", help="Do not open in browser automatically"
    )
    parser.add_argument("--dml", action="store_true", help="torch_dml")
    options = parser.parse_args(argv)
    if not 0 <= options.listen_port <= 65535:
        options.listen_port = 7865
    return options
