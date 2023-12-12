# stdlib
from typing import List

# tls_crawler relative
from .base import BasicStats


class PacketTime(BasicStats):
    """This class extracts features related to the Packet Times."""

    count = 0

    def __init__(self, packets: List) -> None:
        super().__init__()
        self.packets = packets
        PacketTime.count += 1

    def get_observed_data(self) -> List:
        return self._get_packet_times()

    def _get_packet_times(self) -> List:
        """Gets a list of the times of the packets on a flow

        Returns:
            A list of the packet times.

        """
        first_packet_time = float(self.packets[0][0].frame_info.time_epoch)
        packet_times = [
            float(packet.frame_info.time_epoch) - first_packet_time
            for packet, _ in self.packets
        ]
        return packet_times

    def relative_time_list(self) -> List:
        relative_time_list: List[float] = []
        packet_times = self._get_packet_times()
        for index, time in enumerate(packet_times):
            if index == 0:
                relative_time_list.append(0)
            elif index < len(packet_times):
                relative_time_list.append(float(time - packet_times[index - 1]))
            elif index < 50:
                relative_time_list.append(0)
            else:
                break

        return relative_time_list

    def get_time_stamp(self) -> float:
        """Returns the timestamp of the first packet."""
        return float(self.packets[0][0].frame_info.time_epoch)

    def get_duration(self) -> float:
        """Calculates the duration of a network flow.

        Returns:
            The duration of a network flow.

        """

        return max(self._get_packet_times()) - min(self._get_packet_times())
