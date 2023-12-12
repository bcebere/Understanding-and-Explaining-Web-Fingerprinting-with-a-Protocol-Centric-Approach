# stdlib
from enum import Enum
import hashlib
from typing import Any, Dict, List, Tuple

# third party
import pandas as pd
import tldextract

# tls_crawler absolute
from tls_crawler.processing.features.app_layer import AppLayerFeatures
from tls_crawler.processing.features.context import packet_flow_key
from tls_crawler.processing.features.flow_bytes import FlowBytes
from tls_crawler.processing.features.packet_length import PacketLength
from tls_crawler.processing.features.packet_time import PacketTime
from tls_crawler.processing.features.response_time import ResponseTime
from tls_crawler.processing.features.time_series import TimeSeries


class Flow:
    """This class summarizes the values of the features of the network flows"""

    def __init__(
        self,
        packet: Any,
        direction: Enum,
        dns_reverse_cache: dict = {},
        certificates: dict = {},
        buffer_tcp: bool = True,
    ) -> None:
        """This method initializes an object from the Flow class.

        Args:
            packet (Any): A packet from the network.
            direction (Enum): The direction the packet is going ove the wire.
        """

        (
            self.dest_ip,
            self.src_ip,
            self.src_port,
            self.dest_port,
        ) = packet_flow_key.get_packet_flow_key(packet, direction)

        self.packets: List[Any] = []
        self.latest_timestamp = 0.0
        self.start_timestamp = 0.0
        self.dns_reverse_cache = dns_reverse_cache
        self.certificates = certificates
        self.buffer_tcp = buffer_tcp

    def server_name(self) -> str:
        transport_features = AppLayerFeatures(self.packets)
        raw_app_data = transport_features.data()

        if "sni" in raw_app_data:
            domain = raw_app_data["sni"]
        elif self.src_ip in self.dns_reverse_cache:
            domain = self.dns_reverse_cache[self.src_ip]
        elif self.dest_ip in self.dns_reverse_cache:
            domain = self.dns_reverse_cache[self.dest_ip]
        else:
            domain = ""

        return domain

    def proto(self) -> str:
        transport_features = AppLayerFeatures(self.packets)
        raw_app_data = transport_features.data()
        return raw_app_data["app_proto"] if "app_proto" in raw_app_data else ""

    def app_info(self) -> dict:
        domain = self.server_name()
        app_proto = self.proto()

        domain_tokens = tldextract.extract(domain)

        app_data: Dict[str, str] = {}
        app_data["domain"] = domain
        app_data["app_proto"] = app_proto
        app_data["subject_CN"] = ""
        app_data["issuer_CN"] = ""

        if domain in self.certificates:
            if "subject" in self.certificates[domain]:
                if "CN" in self.certificates[domain]["subject"]:
                    app_data["subject_CN"] = self.certificates[domain]["subject"]["CN"]
            if "issuer" in self.certificates[domain]:
                if "CN" in self.certificates[domain]["issuer"]:
                    app_data["issuer_CN"] = self.certificates[domain]["issuer"]["CN"]
                if "O" in self.certificates[domain]["issuer"]:
                    app_data["issuer_O"] = self.certificates[domain]["issuer"]["O"]
                if "C" in self.certificates[domain]["issuer"]:
                    app_data["issuer_C"] = self.certificates[domain]["issuer"]["C"]
        app_data["domain_sub"] = domain_tokens.subdomain
        app_data["domain_token"] = domain_tokens.domain
        app_data["domain_suffix"] = domain_tokens.suffix

        return app_data

    def flow_ley(self) -> str:
        keys = (self.src_ip, self.dest_ip, self.src_port, self.dest_port)
        flow_hash = hashlib.sha1()
        for key in keys:
            flow_hash.update(key.encode())
        return flow_hash.hexdigest()

    def static_stats(self) -> dict:
        """This method obtains the values of the features extracted from each flow.

        Note:
            Only some of the network data plays well together in this list.
            Time-to-live values, window values, and flags cause the data to separate out too
            much.

        Returns:
           list: returns a List of values to be outputted into a csv file.

        """

        flow_bytes = FlowBytes(self.packets)
        packet_length_raw_sent = PacketLength(self.packets, strategy="raw_sent")
        packet_length_raw_recv = PacketLength(self.packets, strategy="raw_recv")
        packet_length_raw_all = PacketLength(self.packets, strategy="raw_all")
        packet_length_aggr_sent = PacketLength(self.packets, strategy="aggr_sent")
        packet_length_aggr_recv = PacketLength(self.packets, strategy="aggr_recv")
        packet_length_aggr_all = PacketLength(self.packets, strategy="aggr_all")
        packet_time = PacketTime(self.packets)
        response = ResponseTime(self.packets)

        app_data = self.app_info()

        data = {
            # Basic IP information
            "source_ip": self.src_ip,
            "destination_ip": self.dest_ip,
            "source_port": self.src_port,
            "destination_port": self.dest_port,
            # Basic information from packet times
            "timestamp": packet_time.get_time_stamp(),
            "duration": packet_time.get_duration(),
            "packet_cnt": len(self.packets),
            # Information from the amount of bytes
            "flow_bytes_sent": flow_bytes.get_bytes_sent(),
            "flow_sent_rate": flow_bytes.get_sent_rate(),
            "flow_bytes_received": flow_bytes.get_bytes_received(),
            "flow_received_rate": flow_bytes.get_received_rate(),
            # Statistical info obtained from (Raw, Sent) Packet lengths
            "packet_length_std_raw_sent": packet_length_raw_sent.get_std(),
            "packet_length_mean_raw_sent": packet_length_raw_sent.get_mean(),
            "packet_length_median_raw_sent": packet_length_raw_sent.get_median(),
            "packet_length_skew_from_median_raw_sent": packet_length_raw_sent.get_skew(),
            "packet_length_coeff_variation_raw_sent": packet_length_raw_sent.get_cov(),
            # Statistical info obtained from (Raw, Recv) Packet lengths
            "packet_length_std_raw_recv": packet_length_raw_recv.get_std(),
            "packet_length_mean_raw_recv": packet_length_raw_recv.get_mean(),
            "packet_length_median_raw_recv": packet_length_raw_recv.get_median(),
            "packet_length_skew_from_median_raw_recv": packet_length_raw_recv.get_skew(),
            "packet_length_coeff_variation_raw_recv": packet_length_raw_recv.get_cov(),
            # Statistical info obtained from (Raw, All) Packet lengths
            "packet_length_std_raw_all": packet_length_raw_all.get_std(),
            "packet_length_mean_raw_all": packet_length_raw_all.get_mean(),
            "packet_length_median_raw_all": packet_length_raw_all.get_median(),
            "packet_length_skew_from_median_raw_all": packet_length_raw_all.get_skew(),
            "packet_length_coeff_variation_raw_all": packet_length_raw_all.get_cov(),
            # Statistical info obtained from (Aggr, Sent) Packet lengths
            "packet_length_std_aggr_sent": packet_length_aggr_sent.get_std(),
            "packet_length_mean_aggr_sent": packet_length_aggr_sent.get_mean(),
            "packet_length_median_aggr_sent": packet_length_aggr_sent.get_median(),
            "packet_length_skew_from_median_aggr_sent": packet_length_aggr_sent.get_skew(),
            "packet_length_coeff_variation_aggr_sent": packet_length_aggr_sent.get_cov(),
            # Statistical info obtained from (Aggr, Recv) Packet lengths
            "packet_length_std_aggr_recv": packet_length_aggr_recv.get_std(),
            "packet_length_mean_aggr_recv": packet_length_aggr_recv.get_mean(),
            "packet_length_median_aggr_recv": packet_length_aggr_recv.get_median(),
            "packet_length_skew_from_median_aggr_recv": packet_length_aggr_recv.get_skew(),
            "packet_length_coeff_variation_aggr_recv": packet_length_aggr_recv.get_cov(),
            # Statistical info obtained from (Aggr, All) Packet lengths
            "packet_length_std_aggr_all": packet_length_aggr_all.get_std(),
            "packet_length_mean_aggr_all": packet_length_aggr_all.get_mean(),
            "packet_length_median_aggr_all": packet_length_aggr_all.get_median(),
            "packet_length_skew_from_median_aggr_all": packet_length_aggr_all.get_skew(),
            "packet_length_coeff_variation_aggr_all": packet_length_aggr_all.get_cov(),
            # Statistical info  obtained from Packet times
            "packet_time_std": packet_time.get_std(),
            "packet_time_mean": packet_time.get_mean(),
            "packet_time_median": packet_time.get_median(),
            "packet_time_skew_from_median": packet_time.get_skew(),
            "packet_time_coeff_variation": packet_time.get_cov(),
            # Response Time
            "response_time_std": response.get_std(),
            "response_time_mean": response.get_mean(),
            "response_time_median": response.get_median(),
            "response_time_skew_from_median": response.get_skew(),
            "response_time_coeff_variation": response.get_cov(),
        }
        for key in app_data:
            data[key] = app_data[key]

        return data

    def temporal_stats(self) -> Tuple[dict, pd.DataFrame]:
        flow_key = self.flow_ley()
        app_info = self.app_info()

        static_data = {
            # Basic IP information
            "id": flow_key,
            "source_ip": self.src_ip,
            "destination_ip": self.dest_ip,
            "source_port": self.src_port,
            "destination_port": self.dest_port,
        }
        for key in app_info:
            static_data[key] = app_info[key]

        temporal_data = TimeSeries(self.packets, buffer_tcp=self.buffer_tcp).data()

        temporal_cols = ["id"] + list(temporal_data.columns)

        temporal_data["id"] = flow_key
        temporal_data = temporal_data[temporal_cols]

        return static_data, temporal_data

    def add_packet(self, packet: Any, direction: Enum) -> None:
        """Adds a packet to the current list of packets.

        Args:
            packet: Packet to be added to a flow
            direction: The direction the packet is going in that flow

        """
        self.packets.append((packet, direction))

        self.latest_timestamp = max(
            [float(packet.frame_info.time_epoch), self.latest_timestamp]
        )

        if self.start_timestamp == 0:
            self.start_timestamp = float(packet.frame_info.time_epoch)

    @property
    def duration(self) -> float:
        return self.latest_timestamp - self.start_timestamp

    def __len__(self) -> int:
        return len(self.packets)
