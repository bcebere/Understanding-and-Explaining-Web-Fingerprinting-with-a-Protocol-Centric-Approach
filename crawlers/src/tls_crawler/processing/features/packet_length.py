# stdlib
from typing import List, Optional

# third party
# tls_crawler absolute
from tls_crawler.processing.features.context.packet_direction import PacketDirection

# tls_crawler relative
from .base import BasicStats


class PacketLength(BasicStats):
    """This class extracts features related to the Packet Lengths.

    Attributes:
        mean_count (int): The row number.
        grand_total (float): The cummulative total of the means.

    """

    mean_count = 0
    grand_total = 0

    def __init__(
        self,
        packets: List,
        strategy: str,  # raw_all, raw_sent, raw_recv, aggr_all, aggr_sent, aggr_recv,
    ) -> None:
        super().__init__()
        assert strategy in [
            "raw_all",
            "raw_sent",
            "raw_recv",
            "aggr_all",
            "aggr_sent",
            "aggr_recv",
        ]

        self.strategy = strategy
        self.packets = packets
        self.interesting_packets: Optional[List] = None

    def get_observed_data(self) -> List:
        if self.interesting_packets is not None:
            return self.interesting_packets

        if self.strategy == "raw_all":
            self.interesting_packets = self.get_raw_packet_length()
        elif self.strategy == "raw_sent":
            self.interesting_packets = self.get_raw_packet_length_sent()
        elif self.strategy == "raw_recv":
            self.interesting_packets = self.get_raw_packet_length_recv()
        elif self.strategy == "aggr_all":
            self.interesting_packets = self.get_aggr_packet_length()
        elif self.strategy == "aggr_recv":
            self.interesting_packets = self.get_aggr_packet_length_recv()
        elif self.strategy == "aggr_sent":
            self.interesting_packets = self.get_aggr_packet_length_sent()
        else:
            raise RuntimeError(f"invalid strategy {self.strategy}")

        return self.interesting_packets

    def get_raw_packet_length(self) -> list:
        """Creates a list of packet lengths.

        Returns:
            packet_lengths (List[int]):

        """
        return [len(packet) for packet, _ in self.packets]

    def get_raw_packet_length_sent(self) -> list:
        """Sent packets.

        Returns:
            packet_lengths (List[int]):

        """
        return [
            len(packet)
            for packet, direction in self.packets
            if direction == PacketDirection.FORWARD
        ]

    def get_raw_packet_length_recv(self) -> list:
        """Recv packets.

        Returns:
            packet_lengths (List[int]):

        """
        return [
            len(packet)
            for packet, direction in self.packets
            if direction == PacketDirection.REVERSE
        ]

    def _aggregated(self) -> list:
        buffered = []
        buff_len = 0
        buff_direction = PacketDirection.FORWARD
        for packet, direction in self.packets:
            if direction != buff_direction:
                if buff_len != 0:
                    buffered.append((buff_len, buff_direction))

                buff_direction = direction
                buff_len = len(packet)
            else:
                buff_len += len(packet)

        if buff_len != 0:
            buffered.append((buff_len, buff_direction))

        return buffered

    def get_aggr_packet_length(self) -> list:
        """Creates a list of packet lengths.

        Returns:
            packet_lengths (List[int]):

        """
        return [packet for packet, _ in self._aggregated()]

    def get_aggr_packet_length_sent(self) -> list:
        """Creates a list of packet lengths.

        Returns:
            packet_lengths (List[int]):

        """
        return [
            packet
            for packet, direction in self._aggregated()
            if direction == PacketDirection.FORWARD
        ]

    def get_aggr_packet_length_recv(self) -> list:
        """Creates a list of packet lengths.

        Returns:
            packet_lengths (List[int]):

        """
        return [
            packet
            for packet, direction in self._aggregated()
            if direction == PacketDirection.REVERSE
        ]
