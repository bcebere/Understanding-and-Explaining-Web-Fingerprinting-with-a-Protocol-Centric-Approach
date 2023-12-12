# stdlib
from datetime import datetime
from pathlib import Path
import socket
import ssl
import time

# third party
import OpenSSL


def retrieve_certificate(host: str, port: int = 443, timeout: int = 1) -> str:
    context = ssl.create_default_context()

    conn = socket.create_connection((host, port), timeout=timeout)

    sock = context.wrap_socket(conn, server_hostname=host)
    sock.settimeout(timeout)

    try:
        der_cert = sock.getpeercert(True)
    finally:
        sock.close()

    if der_cert is None:
        return ""

    return ssl.DER_cert_to_PEM_cert(der_cert)


def certificate(domain: str, workspace: Path = Path("workspace")) -> dict:
    workspace.mkdir(parents=True, exist_ok=True)
    backup = workspace / f"{domain}.cert"
    certificate = None
    if backup.exists():
        with open(backup) as f:
            try:
                certificate = f.read()
            except BaseException:
                pass

    if certificate is None:
        certificate = ""
        try:
            certificate = retrieve_certificate(domain)
        except BaseException:
            pass
        with open(backup, "w") as f:
            f.write(certificate)

    x509 = OpenSSL.crypto.load_certificate(OpenSSL.crypto.FILETYPE_PEM, certificate)  # type: ignore

    raw_subject_data = dict(x509.get_subject().get_components())
    subject_data = {}
    for key in raw_subject_data:
        subject_data[key.decode("utf-8")] = raw_subject_data[key].decode("utf-8")

    raw_issuer = dict(x509.get_issuer().get_components())
    issuer_data = {}
    for key in raw_issuer:
        issuer_data[key.decode("utf-8")] = raw_issuer[key].decode("utf-8")

    return {
        "subject": subject_data,
        "issuer": issuer_data,
        "version": x509.get_version(),
        "not_before": time.mktime(
            datetime.strptime(
                x509.get_notBefore().decode("utf-8"), "%Y%m%d%H%M%SZ"
            ).timetuple()
        ),
        "not_after": time.mktime(
            datetime.strptime(
                x509.get_notAfter().decode("utf-8"), "%Y%m%d%H%M%SZ"
            ).timetuple()
        ),
    }
