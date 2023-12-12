# stdlib
from pathlib import Path

# third party
import pyshark

# tls_crawler absolute
from tls_crawler.processing.flow_session import FlowSession


def process_pcap(
    input_file: Path,
    workspace: Path = Path("workspace"),
    with_certificates: bool = False,
    with_dns: bool = True,
    buffer_tcp: bool = True,
) -> FlowSession:
    input_file = Path(input_file)
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    filtered_cap = pyshark.FileCapture(
        input_file,
        display_filter="tls or dns or quic",
    )

    session = FlowSession(
        filtered_cap,
        name=input_file.stem,
        workspace=workspace,
        with_certificates=with_certificates,
        with_dns=with_dns,
        buffer_tcp=buffer_tcp,
    )

    filtered_cap.close()

    return session
