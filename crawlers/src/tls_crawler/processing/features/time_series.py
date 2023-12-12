# stdlib
from typing import Any, Generator

# third party
import pandas as pd

# tls_crawler absolute
from tls_crawler.processing.features.context.packet_direction import PacketDirection


class Clump:
    """Represents several packets with the same direction with short time between them"""

    def __init__(self, direction: PacketDirection) -> None:
        self.direction = direction
        self.packets = 0
        self.size = 0
        self.latest_timestamp = 0.0
        self.first_timestamp = 0.0

    def add_packet(self, packet: Any) -> None:
        if self.first_timestamp == 0:
            self.first_timestamp = float(packet.frame_info.time_epoch)
        self.packets += 1
        self.size += len(packet)
        self.latest_timestamp = float(packet.frame_info.time_epoch)

    def accepts(self, packet: Any, direction: PacketDirection) -> bool:
        return direction == self.direction

    def duration(self) -> float:
        return self.latest_timestamp - self.first_timestamp


class TimeSeries:
    def __init__(self, packets: list, buffer_tcp: bool = True) -> None:
        self.packets = packets
        self.buffer_tcp = buffer_tcp

    def _clumps(self) -> Generator:
        current_clump = None

        for packet, direction in self.packets:
            if "TLS" not in packet and "QUIC" not in packet:
                continue

            if current_clump is None:
                current_clump = Clump(direction=direction)

            if not current_clump.accepts(packet, direction):
                yield current_clump
                current_clump = Clump(direction=direction)

            if len(packet) < 50:
                continue

            if "TCP" in packet:
                lost_ack = None
                try:
                    lost_ack = packet["TCP"].analysis_ack_lost_segment
                except BaseException:
                    pass

                if lost_ack is not None:
                    raise RuntimeError(
                        f"Drop flow with missing packets srcport = {packet['TCP'].srcport}"
                    )

            current_clump.add_packet(packet)

        if current_clump is not None:
            yield current_clump

    def data(self) -> pd.DataFrame:
        results = []

        if self.buffer_tcp:
            latest_clump_end_timestamp = None

            count = 0
            for c in self._clumps():
                if latest_clump_end_timestamp is None:
                    latest_clump_end_timestamp = c.first_timestamp
                count += 1
                results.append(
                    [
                        float(
                            c.first_timestamp - latest_clump_end_timestamp
                        ),  # inter-arrival duration
                        float(c.duration()),
                        c.size,
                        c.packets,
                        1 if c.direction == PacketDirection.FORWARD else -1,
                    ]
                )
                latest_clump_end_timestamp = c.latest_timestamp
        else:
            latest_timestamp = 0.0

            for packet, direction in self.packets:
                if "TLS" not in packet and "QUIC" not in packet:
                    continue

                if len(packet) < 70:
                    continue

                if "TCP" in packet:
                    lost_ack = None
                    try:
                        lost_ack = packet["TCP"].analysis_ack_lost_segment
                    except BaseException:
                        pass

                    if lost_ack is not None:
                        raise RuntimeError(
                            f"Drop flow with missing packets srcport = {packet['TCP'].srcport}"
                        )

                packet_len = len(packet)
                packet_timestamp = float(packet.frame_info.time_epoch)

                results.append(
                    [
                        float(
                            packet_timestamp - latest_timestamp,
                        ),  # inter-arrival duration
                        0,
                        packet_len,
                        1,
                        1 if direction == PacketDirection.FORWARD else -1,
                    ]
                )
                latest_timestamp = packet_timestamp

        return pd.DataFrame(
            results,
            columns=[
                "relative_timestamp",
                "duration",
                "length",
                "pkt_count",
                "direction",
            ],
        )
