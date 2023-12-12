# stdlib
from typing import List

# third party
# tls_crawler absolute
import tls_crawler.logger as log

# use .field_names for viewing available layer members


class AppLayerFeatures:
    count = 0

    def __init__(self, packets: List) -> None:
        super().__init__()
        self.packets = packets

    def all_snis(self) -> list:
        results = []
        for packet, direction in self.packets:
            if "TLS" in packet:
                try:
                    sni = packet.tls.handshake_extensions_server_name
                    results.append(sni)
                except BaseException:
                    continue
                # print(packet.tls.field_names)
        return results

    def data(self) -> dict:
        features = {}
        for packet, direction in self.packets:
            if "TLS" in packet:
                try:
                    features["sni"] = packet.tls.handshake_extensions_server_name
                    features["app_proto"] = "TLS"
                except BaseException as e:
                    log.debug(f"failed to parse TLS handshake {e}")
                # print(packet.tls.field_names)
                break
            elif "QUIC" in packet:
                # TODO
                features["app_proto"] = "QUIC"
                pass
            elif "DNS" in packet:
                # print(packet["DNS"].field_names)
                features["dns"] = packet["DNS"].qry_name
                features["app_proto"] = "DNS"
        return features
