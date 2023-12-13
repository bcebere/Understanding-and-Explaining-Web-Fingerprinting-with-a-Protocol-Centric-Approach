# stdlib
import glob
import json
from pathlib import Path
from random import shuffle

# third party
# tls_crawler absolute
from tls_crawler.processing import process_pcap


def get_temporal_stats_per_flow(country: str) -> None:
    workspace = Path(f"traces_{country}")
    output = Path(f"datasets_temporal_{country}")
    output.mkdir(parents=True, exist_ok=True)

    files = glob.glob(str(workspace / "*.pcap"))
    shuffle(files)

    for filename in files:
        print("evaluate ", filename, flush=True)
        stem = Path(filename).stem
        output_csv_static = output / f"flow_static_data_{stem}.csv"
        output_csv_temporal = output / f"flow_temporal_data_{stem}.csv"
        file_meta = workspace / f"{stem}.meta"

        assert file_meta.exists()

        if output_csv_static.exists():
            print("already cached", output_csv_static)
            continue

        with open(file_meta) as f:
            meta = json.loads(f.read())

        session = process_pcap(Path(filename))
        static_data, temporal_data = session.temporal_stats_per_flow()

        static_data["use_adblocking"] = meta["use_adblocking"]
        static_data["use_tracking_blocking"] = meta["use_tracking_blocking"]
        static_data["use_user_cookies"] = meta["use_user_cookies"]
        static_data["use_http_caching"] = meta["use_http_caching"]
        static_data["country"] = meta["country"]
        static_data["network_type"] = meta["network_type"]
        static_data["meta_file"] = str(file_meta)
        for header_key in meta["first_server_headers"]:
            static_data[header_key] = meta["first_server_headers"][header_key]
        static_data["user_agent"] = meta["user_agent"]
        static_data["request_path"] = meta["request"]["path"]
        temporal_data["meta_file"] = str(file_meta)

        static_data.to_csv(output_csv_static, index=False)
        temporal_data.to_csv(output_csv_temporal, index=False)


get_temporal_stats_per_flow(country="DE")
