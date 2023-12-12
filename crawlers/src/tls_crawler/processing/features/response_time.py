# stdlib
from typing import Any, List, Optional

# third party
# tls_crawler absolute
from tls_crawler.processing.features.context.packet_direction import PacketDirection

# tls_crawler relative
from .base import BasicStats


class ResponseTime(BasicStats):
    """A summary of features based on the time difference \
       between an outgoing packet and the following response.
    """

    def __init__(self, packets: List) -> None:
        super().__init__()
        self.packets = packets

    def get_observed_data(self) -> List:
        return self._get_dif()

    def _get_dif(self) -> list:
        """Calculates the time difference in seconds between\
           an outgoing packet and the following response packet.

        Returns:
            List[float]: A list of time differences.

        """
        time_diff = []
        temp_packet: Optional[Any] = None
        temp_direction = None
        for packet, direction in self.packets:
            if (
                temp_direction == PacketDirection.FORWARD
                and direction == PacketDirection.REVERSE
            ) and temp_packet is not None:
                time_diff.append(
                    float(packet.frame_info.time_epoch)
                    - float(temp_packet.frame_info.time_epoch)
                )
            temp_packet = packet
            temp_direction = direction
        return time_diff
