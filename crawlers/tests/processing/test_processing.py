# stdlib
from pathlib import Path

# third party
import pandas as pd
import pytest

# tls_crawler absolute
from tls_crawler.processing import process_pcap
from tls_crawler.processing.features.context.packet_direction import PacketDirection
from tls_crawler.processing.features.packet_length import PacketLength


@pytest.mark.parametrize(
    "strategy",
    ["raw_all", "raw_sent", "raw_recv", "aggr_all", "aggr_sent", "aggr_recv"],
)
def test_packet_length(strategy: str) -> None:
    packets = [
        ([1] * 10, PacketDirection.FORWARD),
        ([1] * 31, PacketDirection.REVERSE),
        ([1] * 101, PacketDirection.REVERSE),
        ([1] * 12, PacketDirection.FORWARD),
        ([1] * 14, PacketDirection.FORWARD),
    ]
    lengths = PacketLength(packets, strategy=strategy).get_observed_data()

    if strategy == "raw_all":
        assert lengths == [10, 31, 101, 12, 14]
    elif strategy == "raw_sent":
        assert lengths == [10, 12, 14]
    elif strategy == "raw_recv":
        assert lengths == [31, 101]
    elif strategy == "aggr_all":
        assert lengths == [10, 132, 26]
    elif strategy == "aggr_sent":
        assert lengths == [10, 26]
    elif strategy == "aggr_recv":
        assert lengths == [132]
    else:
        raise RuntimeError()


def test_sanity_static_per_flow() -> None:
    session = process_pcap(Path("test2.pcap"))
    static_stats = session.static_stats_per_flow()

    assert isinstance(static_stats, pd.DataFrame)
    expected_cols = [
        "source_ip",
        "destination_ip",
        "source_port",
        "destination_port",
        "timestamp",
        "duration",
        "packet_cnt",
        "flow_bytes_sent",
        "flow_sent_rate",
        "flow_bytes_received",
        "flow_received_rate",
        "packet_length_std_raw_sent",
        "packet_length_mean_raw_sent",
        "packet_length_median_raw_sent",
        "packet_length_skew_from_median_raw_sent",
        "packet_length_coeff_variation_raw_sent",
        "packet_length_std_raw_recv",
        "packet_length_mean_raw_recv",
        "packet_length_median_raw_recv",
        "packet_length_skew_from_median_raw_recv",
        "packet_length_coeff_variation_raw_recv",
        "packet_length_std_raw_all",
        "packet_length_mean_raw_all",
        "packet_length_median_raw_all",
        "packet_length_skew_from_median_raw_all",
        "packet_length_coeff_variation_raw_all",
        "packet_length_std_aggr_sent",
        "packet_length_mean_aggr_sent",
        "packet_length_median_aggr_sent",
        "packet_length_skew_from_median_aggr_sent",
        "packet_length_coeff_variation_aggr_sent",
        "packet_length_std_aggr_recv",
        "packet_length_mean_aggr_recv",
        "packet_length_median_aggr_recv",
        "packet_length_skew_from_median_aggr_recv",
        "packet_length_coeff_variation_aggr_recv",
        "packet_length_std_aggr_all",
        "packet_length_mean_aggr_all",
        "packet_length_median_aggr_all",
        "packet_length_skew_from_median_aggr_all",
        "packet_length_coeff_variation_aggr_all",
        "response_time_std",
        "response_time_mean",
        "response_time_median",
        "response_time_skew_from_median",
        "response_time_coeff_variation",
        "domain",
        "app_proto",
        "subject_CN",
        "issuer_CN",
        "issuer_O",
        "issuer_C",
        "domain_sub",
        "domain_token",
        "domain_suffix",
    ]

    for col in expected_cols:
        assert col in static_stats


def test_sanity_static_contextual() -> None:
    session = process_pcap(Path("test.pcap"))
    static_stats_ctx = session.static_stats_per_context()
    static_stats_per_flow = session.static_stats_per_flow()
    print("contextual", static_stats_ctx)

    assert isinstance(static_stats_ctx, pd.DataFrame)
    expected_cols = [
        "timestamp",
        "duration",
        "flow_bytes_sent",
        "flow_sent_rate",
        "flow_bytes_received",
        "flow_received_rate",
        "packet_length_std_raw_sent",
        "packet_length_mean_raw_sent",
        "packet_length_median_raw_sent",
        "packet_length_skew_from_median_raw_sent",
        "packet_length_coeff_variation_raw_sent",
        "packet_length_std_raw_recv",
        "packet_length_mean_raw_recv",
        "packet_length_median_raw_recv",
        "packet_length_skew_from_median_raw_recv",
        "packet_length_coeff_variation_raw_recv",
        "packet_length_std_raw_all",
        "packet_length_mean_raw_all",
        "packet_length_median_raw_all",
        "packet_length_skew_from_median_raw_all",
        "packet_length_coeff_variation_raw_all",
        "packet_length_std_aggr_sent",
        "packet_length_mean_aggr_sent",
        "packet_length_median_aggr_sent",
        "packet_length_skew_from_median_aggr_sent",
        "packet_length_coeff_variation_aggr_sent",
        "packet_length_std_aggr_recv",
        "packet_length_mean_aggr_recv",
        "packet_length_median_aggr_recv",
        "packet_length_skew_from_median_aggr_recv",
        "packet_length_coeff_variation_aggr_recv",
        "packet_length_std_aggr_all",
        "packet_length_mean_aggr_all",
        "packet_length_median_aggr_all",
        "packet_length_skew_from_median_aggr_all",
        "packet_length_coeff_variation_aggr_all",
        "packet_time_std",
        "packet_time_mean",
        "packet_time_median",
        "packet_time_skew_from_median",
        "packet_time_coeff_variation",
        "response_time_std",
        "response_time_mean",
        "response_time_median",
        "response_time_skew_from_median",
        "response_time_coeff_variation",
        "request_snis",
    ]

    for col in expected_cols:
        assert col in static_stats_ctx

    assert (
        static_stats_ctx["flow_bytes_sent"].values[0]
        == static_stats_per_flow["flow_bytes_sent"].sum()
    )
    assert (
        static_stats_ctx["flow_bytes_received"].values[0]
        == static_stats_per_flow["flow_bytes_received"].sum()
    )


@pytest.mark.parametrize("buffer_tcp", [False, True])
def test_sanity_temporal_per_flow(buffer_tcp: bool) -> None:
    session = process_pcap(
        Path("http2_debug.pcap"), with_certificates=True, buffer_tcp=buffer_tcp
    )
    static, temporal = session.temporal_stats_per_flow()
    print(temporal)
    expected_temporal_cols = [
        "id",
        "relative_timestamp",
        "duration",
        "length",
        "pkt_count",
        "direction",
    ]
    expected_static_cols = [
        "id",
        "source_ip",
        "destination_ip",
        "source_port",
        "destination_port",
        "domain",
        "app_proto",
        "subject_CN",
        "issuer_CN",
        "issuer_O",
        "issuer_C",
        "domain_sub",
        "domain_token",
        "domain_suffix",
    ]

    for col in expected_temporal_cols:
        assert col in temporal

    for col in expected_static_cols:
        assert col in static

    print(static)
    temporal.to_csv("ua_debug.csv", index=None)


def test_sanity_temporal_per_context() -> None:
    session = process_pcap(Path("test.pcap"))
    static, temporal = session.temporal_stats_per_context()
    expected_temporal_cols = [
        "relative_timestamp",
        "duration",
        "length",
        "pkt_count",
        "direction",
    ]
    expected_static_cols = [
        "duration",
        "flow_bytes_sent",
        "flow_sent_rate",
        "flow_bytes_received",
        "flow_received_rate",
        "packet_length_std_raw_sent",
        "packet_length_mean_raw_sent",
        "packet_length_median_raw_sent",
        "packet_length_skew_from_median_raw_sent",
        "packet_length_coeff_variation_raw_sent",
        "packet_length_std_raw_recv",
        "packet_length_mean_raw_recv",
        "packet_length_median_raw_recv",
        "packet_length_skew_from_median_raw_recv",
        "packet_length_coeff_variation_raw_recv",
        "packet_length_std_raw_all",
        "packet_length_mean_raw_all",
        "packet_length_median_raw_all",
        "packet_length_skew_from_median_raw_all",
        "packet_length_coeff_variation_raw_all",
        "packet_length_std_aggr_sent",
        "packet_length_mean_aggr_sent",
        "packet_length_median_aggr_sent",
        "packet_length_skew_from_median_aggr_sent",
        "packet_length_coeff_variation_aggr_sent",
        "packet_length_std_aggr_recv",
        "packet_length_mean_aggr_recv",
        "packet_length_median_aggr_recv",
        "packet_length_skew_from_median_aggr_recv",
        "packet_length_coeff_variation_aggr_recv",
        "packet_length_std_aggr_all",
        "packet_length_mean_aggr_all",
        "packet_length_median_aggr_all",
        "packet_length_skew_from_median_aggr_all",
        "packet_length_coeff_variation_aggr_all",
        "packet_time_std",
        "packet_time_mean",
        "packet_time_median",
        "packet_time_skew_from_median",
        "packet_time_coeff_variation",
        "response_time_std",
        "response_time_mean",
        "response_time_median",
        "response_time_skew_from_median",
        "response_time_coeff_variation",
    ]

    for col in expected_temporal_cols:
        assert col in temporal

    for col in expected_static_cols:
        assert col in static
