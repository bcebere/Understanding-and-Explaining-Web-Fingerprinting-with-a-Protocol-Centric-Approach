# stdlib
from pathlib import Path
import subprocess
import time
from typing import Any, Generator

# third party
import pytest

OFFSET = 999


@pytest.fixture(autouse=True)
def prepare_docker(request: Any) -> Generator:
    tests_dir = Path(request.node.path).parent.name
    if tests_dir == "crawlers":
        workspace = Path(__file__).parent
        print("Start docker")
        subprocess.check_output(["bash", workspace / "start_docker.sh", f"{OFFSET}"])
        time.sleep(2)
        yield
        print("Stop docker")
        subprocess.check_output(["bash", workspace / "stop_docker.sh", f"{OFFSET}"])
    else:
        yield
