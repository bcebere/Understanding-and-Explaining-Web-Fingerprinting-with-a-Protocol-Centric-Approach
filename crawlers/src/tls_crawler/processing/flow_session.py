# stdlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

# third party
import pandas as pd

# tls_crawler absolute
import tls_crawler.logger as log
from tls_crawler.processing.features.app_layer import AppLayerFeatures
from tls_crawler.processing.features.certificates import certificate
from tls_crawler.processing.features.context.packet_direction import PacketDirection
from tls_crawler.processing.features.context.packet_flow_key import get_packet_flow_key
from tls_crawler.processing.features.flow_bytes import FlowBytes
from tls_crawler.processing.features.packet_length import PacketLength
from tls_crawler.processing.features.packet_time import PacketTime
from tls_crawler.processing.features.response_time import ResponseTime
from tls_crawler.processing.features.time_series import TimeSeries
from tls_crawler.processing.flow import Flow


class FlowSession:
    """Creates a list of network flows."""

    def __init__(
        self,
        packets: List,
        name: str,
        with_certificates: bool = False,
        with_dns: bool = True,
        workspace: Path = Path("workspace"),
        buffer_tcp: bool = False,
    ) -> None:
        self.flows: Dict = {}

        self.dns_reverse_cache: Dict[str, str] = {}
        self.certificates: Dict[str, dict] = {}
        self.with_certificates = with_certificates

        self.interesting_packets: List = []

        self.workspace = workspace
        self.buffer_tcp = buffer_tcp

        self._handle_packets(packets)

    def _handle_packets(self, packets: List) -> None:
        for packet in packets:
            if "DNS" in packet:
                try:
                    self.dns_reverse_cache[packet.dns.a] = packet.dns.qry_name
                except BaseException:
                    continue

                if not self.with_certificates:
                    continue

                try:
                    if packet.dns.qry_name in self.certificates:
                        continue
                    self.certificates[packet.dns.qry_name] = certificate(
                        packet.dns.qry_name, workspace=self.workspace
                    )
                except BaseException as e:
                    log.debug(
                        f"failed to get certificate : domain = {packet.dns.qry_name} error ={e}"
                    )

            else:
                self._on_data_packet(packet)

    def _on_data_packet(self, packet: Any) -> None:
        direction = PacketDirection.FORWARD

        # Creates a key variable to check
        packet_flow_key = get_packet_flow_key(packet, direction)
        flow = self.flows.get(packet_flow_key)

        if flow is None:
            # There might be one of it in reverse
            direction = PacketDirection.REVERSE
            packet_flow_key = get_packet_flow_key(packet, direction)
            flow = self.flows.get(packet_flow_key)

            if flow is None:
                # If no flow exists create a new flow
                direction = PacketDirection.FORWARD
                flow = Flow(
                    packet,
                    direction,
                    dns_reverse_cache=self.dns_reverse_cache,
                    certificates=self.certificates,
                    buffer_tcp=self.buffer_tcp,
                )
                packet_flow_key = get_packet_flow_key(packet, direction)
                self.flows[packet_flow_key] = flow

        flow.add_packet(packet, direction)
        self.interesting_packets.append((packet, direction))

    def get_flows(self) -> Any:
        return self.flows.values()

    def static_stats_per_flow(self) -> pd.DataFrame:
        keys = list(self.flows.keys())
        results = {}
        for k in keys:
            flow = self.flows.get(k)
            if flow is None:
                continue
            if len(flow) <= 1:
                continue

            try:
                results[k] = flow.static_stats()
            except BaseException:
                continue

        return pd.DataFrame(results).T.reset_index(drop=True)

    def temporal_stats_per_flow(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        keys = list(self.flows.keys())
        static = {}
        temporal = []
        for k in keys:
            flow = self.flows.get(k)
            if flow is None:
                continue
            if len(flow) <= 1:
                continue

            try:
                local_static, local_temporal = flow.temporal_stats()
                static[k] = local_static
                temporal.append(local_temporal)
            except BaseException as e:
                print("failed to process time series", e)
                continue

        if len(temporal) == 0:
            return pd.DataFrame(), pd.DataFrame()

        return pd.DataFrame(static).T.reset_index(drop=True), pd.concat(
            temporal, ignore_index=True
        )

    def static_stats_per_context(self) -> pd.DataFrame:
        print("static_stats_per_context", len(self.interesting_packets))
        if len(self.interesting_packets) == 0:
            return pd.DataFrame([])

        flow_bytes = FlowBytes(self.interesting_packets)
        packet_length_raw_sent = PacketLength(
            self.interesting_packets, strategy="raw_sent"
        )
        packet_length_raw_recv = PacketLength(
            self.interesting_packets, strategy="raw_recv"
        )
        packet_length_raw_all = PacketLength(
            self.interesting_packets, strategy="raw_all"
        )
        packet_length_aggr_sent = PacketLength(
            self.interesting_packets, strategy="aggr_sent"
        )
        packet_length_aggr_recv = PacketLength(
            self.interesting_packets, strategy="aggr_recv"
        )
        packet_length_aggr_all = PacketLength(
            self.interesting_packets, strategy="aggr_all"
        )
        packet_time = PacketTime(self.interesting_packets)
        response = ResponseTime(self.interesting_packets)
        transport_features = AppLayerFeatures(self.interesting_packets)

        results = {
            # Basic information from packet times
            "timestamp": [packet_time.get_time_stamp()],
            "duration": [packet_time.get_duration()],
            # Information from the amount of bytes
            "flow_bytes_sent": [flow_bytes.get_bytes_sent()],
            "flow_sent_rate": [flow_bytes.get_sent_rate()],
            "flow_bytes_received": [flow_bytes.get_bytes_received()],
            "flow_received_rate": [flow_bytes.get_received_rate()],
            # Statistical info obtained from (Raw, Sent) Packet lengths
            "packet_length_std_raw_sent": [packet_length_raw_sent.get_std()],
            "packet_length_mean_raw_sent": [packet_length_raw_sent.get_mean()],
            "packet_length_median_raw_sent": [packet_length_raw_sent.get_median()],
            "packet_length_skew_from_median_raw_sent": [
                packet_length_raw_sent.get_skew()
            ],
            "packet_length_coeff_variation_raw_sent": [
                packet_length_raw_sent.get_cov()
            ],
            # Statistical info obtained from (Raw, Recv) Packet lengths
            "packet_length_std_raw_recv": [packet_length_raw_recv.get_std()],
            "packet_length_mean_raw_recv": [packet_length_raw_recv.get_mean()],
            "packet_length_median_raw_recv": [packet_length_raw_recv.get_median()],
            "packet_length_skew_from_median_raw_recv": [
                packet_length_raw_recv.get_skew()
            ],
            "packet_length_coeff_variation_raw_recv": [
                packet_length_raw_recv.get_cov()
            ],
            # Statistical info obtained from (Raw, All) Packet lengths
            "packet_length_std_raw_all": [packet_length_raw_all.get_std()],
            "packet_length_mean_raw_all": [packet_length_raw_all.get_mean()],
            "packet_length_median_raw_all": [packet_length_raw_all.get_median()],
            "packet_length_skew_from_median_raw_all": [
                packet_length_raw_all.get_skew()
            ],
            "packet_length_coeff_variation_raw_all": [packet_length_raw_all.get_cov()],
            # Statistical info obtained from (Aggr, Sent) Packet lengths
            "packet_length_std_aggr_sent": [packet_length_aggr_sent.get_std()],
            "packet_length_mean_aggr_sent": [packet_length_aggr_sent.get_mean()],
            "packet_length_median_aggr_sent": [packet_length_aggr_sent.get_median()],
            "packet_length_skew_from_median_aggr_sent": [
                packet_length_aggr_sent.get_skew()
            ],
            "packet_length_coeff_variation_aggr_sent": [
                packet_length_aggr_sent.get_cov()
            ],
            # Statistical info obtained from (Aggr, Recv) Packet lengths
            "packet_length_std_aggr_recv": [packet_length_aggr_recv.get_std()],
            "packet_length_mean_aggr_recv": [packet_length_aggr_recv.get_mean()],
            "packet_length_median_aggr_recv": [packet_length_aggr_recv.get_median()],
            "packet_length_skew_from_median_aggr_recv": [
                packet_length_aggr_recv.get_skew()
            ],
            "packet_length_coeff_variation_aggr_recv": [
                packet_length_aggr_recv.get_cov()
            ],
            # Statistical info obtained from (Aggr, All) Packet lengths
            "packet_length_std_aggr_all": [packet_length_aggr_all.get_std()],
            "packet_length_mean_aggr_all": [packet_length_aggr_all.get_mean()],
            "packet_length_median_aggr_all": [packet_length_aggr_all.get_median()],
            "packet_length_skew_from_median_aggr_all": [
                packet_length_aggr_all.get_skew()
            ],
            "packet_length_coeff_variation_aggr_all": [
                packet_length_aggr_all.get_cov()
            ],
            # Statistical info  obtained from Packet times
            "packet_time_std": [packet_time.get_std()],
            "packet_time_mean": [packet_time.get_mean()],
            "packet_time_median": [packet_time.get_median()],
            "packet_time_skew_from_median": [packet_time.get_skew()],
            "packet_time_coeff_variation": [packet_time.get_cov()],
            # Response Time
            "response_time_std": [response.get_std()],
            "response_time_mean": [response.get_mean()],
            "response_time_median": [response.get_median()],
            "response_time_skew_from_median": [response.get_skew()],
            "response_time_coeff_variation": [response.get_cov()],
            "request_snis": [transport_features.all_snis()],
        }
        return pd.DataFrame(results).reset_index(drop=True)

    def temporal_stats_per_context(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if len(self.interesting_packets) == 0:
            return pd.DataFrame([]), pd.DataFrame([])

        static_results = self.static_stats_per_context()
        temporal_results = TimeSeries(self.interesting_packets).data()

        return pd.DataFrame(static_results).reset_index(drop=True), temporal_results
