# stdlib
from enum import Enum
from typing import Any

# third party
# tls_crawler absolute
from tls_crawler.processing.features.context.packet_direction import PacketDirection


def get_packet_flow_key(packet: Any, direction: Enum) -> tuple:
    """Creates a key signature for a packet.

    Summary:
        Creates a key signature for a packet so it can be
        assigned to a flow.

    Args:
        packet: A network packet
        direction: The direction of a packet

    Returns:
        A tuple of the String IPv4 addresses of the destination,
        the source port as an int,
        the time to live value,
        the window size, and
        TCP flags.

    """
    if "TCP" in packet:
        protocol = "TCP"
    elif "UDP" in packet:
        protocol = "UDP"
    else:
        raise Exception("Only TCP protocols are supported.")

    if direction == PacketDirection.FORWARD:
        dest_ip = packet["IP"].dst
        src_ip = packet["IP"].src
        src_port = packet[protocol].srcport
        dest_port = packet[protocol].dstport
    else:
        dest_ip = packet["IP"].src
        src_ip = packet["IP"].dst
        src_port = packet[protocol].dstport
        dest_port = packet[protocol].srcport

    return dest_ip, src_ip, src_port, dest_port
