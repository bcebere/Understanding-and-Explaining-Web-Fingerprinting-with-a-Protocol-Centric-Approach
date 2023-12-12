# stdlib
from typing import List

# third party
# tls_crawler absolute
from tls_crawler.processing.features.context.packet_direction import PacketDirection
from tls_crawler.processing.features.packet_time import PacketTime


class FlowBytes:
    """Extracts features from the traffic related to the bytes in a flow"""

    def __init__(self, packets: List) -> None:
        self.packets = packets

    def direction_list(self) -> list:
        """Returns a list of the directions of the first 50 packets in a flow.

        Return:
            list with packet directions.

        """
        direction_list = [
            (i, direction.name)[1]
            for (i, (packet, direction)) in enumerate(self.packets)
            if i < 50
        ]
        return direction_list

    def get_bytes_sent(self) -> int:
        """Calculates the amount bytes sent from the machine being used to run DoHlyzer.

        Returns:
            int: The amount of bytes.

        """
        return sum(
            len(packet)
            for packet, direction in self.packets
            if direction == PacketDirection.FORWARD
        )

    def get_sent_rate(self) -> float:
        """Calculates the rate of the bytes being sent in the current flow.

        Returns:
            float: The bytes/sec sent.

        """
        sent = self.get_bytes_sent()
        duration = PacketTime(self.packets).get_duration()

        if duration == 0:
            rate = -1.0
        else:
            rate = sent / duration

        return rate

    def get_bytes_received(self) -> int:
        """Calculates the amount bytes received.

        Returns:
            int: The amount of bytes.

        """
        packets = self.packets

        return sum(
            len(packet)
            for packet, direction in packets
            if direction == PacketDirection.REVERSE
        )

    def get_received_rate(self) -> float:
        """Calculates the rate of the bytes being received in the current flow.

        Returns:
            float: The bytes/sec received.

        """
        received = self.get_bytes_received()
        duration = PacketTime(self.packets).get_duration()

        if duration == 0:
            rate = -1.0
        else:
            rate = received / duration

        return rate
