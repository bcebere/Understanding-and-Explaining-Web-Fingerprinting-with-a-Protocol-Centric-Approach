# third party
import pytest

# tls_crawler absolute
from tls_crawler import load_page_in_firefox

OFFSET = 999
REMOTE_PORT = 2 * OFFSET + 4445


@pytest.mark.parametrize("use_adblocking", [False, True])
def test_page_load_in_firefox_ok_adblocking(
    use_adblocking: bool,
) -> None:
    PAGE = "https://www.google.com/"
    traces = load_page_in_firefox(
        page_name=PAGE,
        use_adblocking=use_adblocking,
        http_caching_repeats=1,
        startup_wait_sec=1,
        page_load_timeout_sec=5,
        repeats=2,
        docker_name=f"selenium-firefox{OFFSET}",
        docker_iface=f"veth_firefox{OFFSET}",
        should_trace=False,
        bpf_filter="tcp",
        country="RO",
        network_type="mobile_data",
        remote_port=REMOTE_PORT,
    )
    assert len(traces) == 2
    for trace in traces:
        assert trace["page_body"] is not None
        assert trace["stacktrace"] is None
        assert trace["metadata"]["engine"] == f"selenium-firefox{OFFSET}"
        assert trace["metadata"]["use_adblocking"] == use_adblocking
        assert trace["metadata"]["http_caching_repeats"] == 1
        assert trace["metadata"]["country"] == "RO"
        assert trace["metadata"]["network_type"] == "mobile_data"
        assert trace["metadata"]["bpf_filter"] == "tcp"
        assert trace["metadata"]["timestamp"] > 0
        assert len(trace["metadata"]["browser_trace"]) > 0


@pytest.mark.parametrize("use_adblocking", [False, True])
def test_page_load_in_firefox_fail_adblocking(
    use_adblocking: bool,
) -> None:
    PAGE = "https://this.domain.does.not.exist/"
    traces = load_page_in_firefox(
        page_name=PAGE,
        use_adblocking=use_adblocking,
        http_caching_repeats=1,
        startup_wait_sec=1,
        page_load_timeout_sec=5,
        repeats=2,
        docker_name=f"selenium-firefox{OFFSET}",
        docker_iface=f"veth_firefox{OFFSET}",
        should_trace=False,
        bpf_filter="tcp",
        country="RO",
        network_type="mobile_data",
        remote_port=REMOTE_PORT,
    )
    assert len(traces) == 1
    for trace in traces:
        assert trace["page_body"] is None
        assert trace["stacktrace"] is not None
        assert trace["metadata"]["engine"] == f"selenium-firefox{OFFSET}"
        assert trace["metadata"]["use_adblocking"] == use_adblocking
        assert trace["metadata"]["http_caching_repeats"] == 1
        assert trace["metadata"]["country"] == "RO"
        assert trace["metadata"]["network_type"] == "mobile_data"
        assert trace["metadata"]["bpf_filter"] == "tcp"
        assert trace["metadata"]["timestamp"] > 0
        assert len(trace["metadata"]["browser_trace"]) == 0


@pytest.mark.parametrize("use_tracking_blocking", [False, True])
def test_page_load_in_firefox_ok_tracking(
    use_tracking_blocking: bool,
) -> None:
    PAGE = "https://www.google.com/"
    traces = load_page_in_firefox(
        page_name=PAGE,
        use_tracking_blocking=use_tracking_blocking,
        http_caching_repeats=1,
        startup_wait_sec=1,
        page_load_timeout_sec=5,
        repeats=2,
        docker_name=f"selenium-firefox{OFFSET}",
        docker_iface=f"veth_firefox{OFFSET}",
        should_trace=False,
        bpf_filter="tcp",
        country="RO",
        network_type="mobile_data",
        remote_port=REMOTE_PORT,
    )
    assert len(traces) == 2
    for trace in traces:
        assert trace["page_body"] is not None
        assert trace["stacktrace"] is None
        assert trace["metadata"]["engine"] == f"selenium-firefox{OFFSET}"
        assert trace["metadata"]["use_tracking_blocking"] == use_tracking_blocking
        assert trace["metadata"]["http_caching_repeats"] == 1
        assert trace["metadata"]["country"] == "RO"
        assert trace["metadata"]["network_type"] == "mobile_data"
        assert trace["metadata"]["bpf_filter"] == "tcp"
        assert trace["metadata"]["timestamp"] > 0
        assert len(trace["metadata"]["browser_trace"]) > 0


@pytest.mark.parametrize("use_tracking_blocking", [False, True])
def test_page_load_in_firefox_fail_tracking(
    use_tracking_blocking: bool,
) -> None:
    PAGE = "https://this.domain.does.not.exist/"
    traces = load_page_in_firefox(
        page_name=PAGE,
        use_tracking_blocking=use_tracking_blocking,
        http_caching_repeats=1,
        startup_wait_sec=1,
        page_load_timeout_sec=5,
        repeats=2,
        docker_name=f"selenium-firefox{OFFSET}",
        docker_iface=f"veth_firefox{OFFSET}",
        should_trace=False,
        bpf_filter="tcp",
        country="RO",
        network_type="mobile_data",
        remote_port=REMOTE_PORT,
    )
    assert len(traces) == 1
    for trace in traces:
        assert trace["page_body"] is None
        assert trace["stacktrace"] is not None
        assert trace["metadata"]["engine"] == f"selenium-firefox{OFFSET}"
        assert trace["metadata"]["use_tracking_blocking"] == use_tracking_blocking
        assert trace["metadata"]["http_caching_repeats"] == 1
        assert trace["metadata"]["country"] == "RO"
        assert trace["metadata"]["network_type"] == "mobile_data"
        assert trace["metadata"]["bpf_filter"] == "tcp"
        assert trace["metadata"]["timestamp"] > 0
        assert len(trace["metadata"]["browser_trace"]) == 0


@pytest.mark.parametrize("use_http_caching", [False, True])
def test_page_load_in_firefox_ok_caching(
    use_http_caching: bool,
) -> None:
    PAGE = "https://www.google.com/"
    traces = load_page_in_firefox(
        page_name=PAGE,
        use_http_caching=use_http_caching,
        http_caching_repeats=1,
        startup_wait_sec=1,
        page_load_timeout_sec=5,
        repeats=2,
        docker_name=f"selenium-firefox{OFFSET}",
        docker_iface=f"veth_firefox{OFFSET}",
        should_trace=False,
        bpf_filter="tcp",
        country="RO",
        network_type="mobile_data",
        remote_port=REMOTE_PORT,
    )
    assert len(traces) == 2
    for trace in traces:
        assert trace["page_body"] is not None
        assert trace["stacktrace"] is None
        assert trace["metadata"]["engine"] == f"selenium-firefox{OFFSET}"
        assert trace["metadata"]["use_http_caching"] == use_http_caching
        assert trace["metadata"]["http_caching_repeats"] == 1
        assert trace["metadata"]["country"] == "RO"
        assert trace["metadata"]["network_type"] == "mobile_data"
        assert trace["metadata"]["bpf_filter"] == "tcp"
        assert trace["metadata"]["timestamp"] > 0
        assert len(trace["metadata"]["browser_trace"]) > 0


@pytest.mark.parametrize("use_http_caching", [False, True])
def test_page_load_in_firefox_fail_caching(
    use_http_caching: bool,
) -> None:
    PAGE = "https://this.domain.does.not.exist/"
    traces = load_page_in_firefox(
        page_name=PAGE,
        use_http_caching=use_http_caching,
        http_caching_repeats=1,
        startup_wait_sec=1,
        page_load_timeout_sec=5,
        repeats=2,
        docker_name=f"selenium-firefox{OFFSET}",
        docker_iface=f"veth_firefox{OFFSET}",
        should_trace=False,
        bpf_filter="tcp",
        country="RO",
        network_type="mobile_data",
        remote_port=REMOTE_PORT,
    )
    assert len(traces) == 1
    for trace in traces:
        assert trace["page_body"] is None
        assert trace["stacktrace"] is not None
        assert trace["metadata"]["engine"] == f"selenium-firefox{OFFSET}"
        assert trace["metadata"]["use_http_caching"] == use_http_caching
        assert trace["metadata"]["http_caching_repeats"] == 1
        assert trace["metadata"]["country"] == "RO"
        assert trace["metadata"]["network_type"] == "mobile_data"
        assert trace["metadata"]["bpf_filter"] == "tcp"
        assert trace["metadata"]["timestamp"] > 0
        assert len(trace["metadata"]["browser_trace"]) == 0


@pytest.mark.parametrize("use_user_cookies", [False, True])
def test_page_load_in_firefox_ok_cookies(
    use_user_cookies: bool,
) -> None:
    PAGE = "https://www.google.com/"
    traces = load_page_in_firefox(
        page_name=PAGE,
        use_user_cookies=use_user_cookies,
        http_caching_repeats=1,
        startup_wait_sec=1,
        page_load_timeout_sec=5,
        repeats=2,
        docker_name=f"selenium-firefox{OFFSET}",
        docker_iface=f"veth_firefox{OFFSET}",
        should_trace=False,
        bpf_filter="tcp",
        country="RO",
        network_type="mobile_data",
        remote_port=REMOTE_PORT,
    )
    assert len(traces) == 2
    for trace in traces:
        assert trace["page_body"] is not None
        assert trace["stacktrace"] is None
        assert trace["metadata"]["engine"] == f"selenium-firefox{OFFSET}"
        assert trace["metadata"]["use_user_cookies"] == use_user_cookies
        assert trace["metadata"]["http_caching_repeats"] == 1
        assert trace["metadata"]["country"] == "RO"
        assert trace["metadata"]["network_type"] == "mobile_data"
        assert trace["metadata"]["bpf_filter"] == "tcp"
        assert trace["metadata"]["timestamp"] > 0
        assert len(trace["metadata"]["browser_trace"]) > 0


@pytest.mark.parametrize("use_user_cookies", [False, True])
def test_page_load_in_firefox_fail_cookies(
    use_user_cookies: bool,
) -> None:
    PAGE = "https://wthis.domain.doesnot.exist/"
    traces = load_page_in_firefox(
        page_name=PAGE,
        use_user_cookies=use_user_cookies,
        http_caching_repeats=1,
        startup_wait_sec=1,
        page_load_timeout_sec=5,
        repeats=2,
        docker_name=f"selenium-firefox{OFFSET}",
        docker_iface=f"veth_firefox{OFFSET}",
        should_trace=False,
        bpf_filter="tcp",
        country="RO",
        network_type="mobile_data",
        remote_port=REMOTE_PORT,
    )
    assert len(traces) == 1
    for trace in traces:
        assert trace["page_body"] is None
        assert trace["stacktrace"] is not None
        assert trace["metadata"]["engine"] == f"selenium-firefox{OFFSET}"
        assert trace["metadata"]["use_user_cookies"] == use_user_cookies
        assert trace["metadata"]["http_caching_repeats"] == 1
        assert trace["metadata"]["country"] == "RO"
        assert trace["metadata"]["network_type"] == "mobile_data"
        assert trace["metadata"]["bpf_filter"] == "tcp"
        assert trace["metadata"]["timestamp"] > 0
        assert len(trace["metadata"]["browser_trace"]) == 0


@pytest.mark.parametrize("use_images", [True, False])
def test_page_load_in_firefox_ok_images(
    use_images: bool,
) -> None:
    PAGE = "https://www.google.com/"
    traces = load_page_in_firefox(
        page_name=PAGE,
        use_images=use_images,
        http_caching_repeats=1,
        startup_wait_sec=1,
        page_load_timeout_sec=5,
        repeats=2,
        docker_name=f"selenium-firefox{OFFSET}",
        docker_iface=f"veth_firefox{OFFSET}",
        should_trace=False,
        bpf_filter="tcp",
        country="RO",
        network_type="mobile_data",
        remote_port=REMOTE_PORT,
    )
    assert len(traces) == 2
    for trace in traces:
        assert trace["page_body"] is not None
        assert trace["stacktrace"] is None
        assert trace["metadata"]["engine"] == f"selenium-firefox{OFFSET}"
        assert trace["metadata"]["use_images"] == use_images
        assert trace["metadata"]["http_caching_repeats"] == 1
        assert trace["metadata"]["country"] == "RO"
        assert trace["metadata"]["network_type"] == "mobile_data"
        assert trace["metadata"]["bpf_filter"] == "tcp"
        assert trace["metadata"]["timestamp"] > 0
        assert len(trace["metadata"]["browser_trace"]) > 0


@pytest.mark.parametrize("use_images", [True, False])
def test_page_load_in_firefox_fail_images(
    use_images: bool,
) -> None:
    PAGE = "https://this.domain.does.not.exist/"
    traces = load_page_in_firefox(
        page_name=PAGE,
        use_images=use_images,
        http_caching_repeats=1,
        startup_wait_sec=1,
        page_load_timeout_sec=5,
        repeats=2,
        docker_name=f"selenium-firefox{OFFSET}",
        docker_iface=f"veth_firefox{OFFSET}",
        should_trace=False,
        bpf_filter="tcp",
        country="RO",
        network_type="mobile_data",
        remote_port=REMOTE_PORT,
    )
    assert len(traces) == 1
    for trace in traces:
        assert trace["page_body"] is None
        assert trace["stacktrace"] is not None
        assert trace["metadata"]["engine"] == f"selenium-firefox{OFFSET}"
        assert trace["metadata"]["use_images"] == use_images
        assert trace["metadata"]["http_caching_repeats"] == 1
        assert trace["metadata"]["country"] == "RO"
        assert trace["metadata"]["network_type"] == "mobile_data"
        assert trace["metadata"]["bpf_filter"] == "tcp"
        assert trace["metadata"]["timestamp"] > 0
        assert len(trace["metadata"]["browser_trace"]) == 0


@pytest.mark.parametrize("use_dns_cache", [True, False])
def test_page_load_in_firefox_ok_dns_cache(
    use_dns_cache: bool,
) -> None:
    PAGE = "https://www.google.com/"
    traces = load_page_in_firefox(
        page_name=PAGE,
        use_dns_cache=use_dns_cache,
        http_caching_repeats=1,
        startup_wait_sec=1,
        page_load_timeout_sec=5,
        repeats=2,
        docker_name=f"selenium-firefox{OFFSET}",
        docker_iface=f"veth_firefox{OFFSET}",
        should_trace=False,
        bpf_filter="tcp",
        country="RO",
        network_type="mobile_data",
        remote_port=REMOTE_PORT,
    )
    assert len(traces) == 2
    for trace in traces:
        assert trace["page_body"] is not None
        assert trace["stacktrace"] is None
        assert trace["metadata"]["engine"] == f"selenium-firefox{OFFSET}"
        assert trace["metadata"]["use_dns_cache"] == use_dns_cache
        assert trace["metadata"]["http_caching_repeats"] == 1
        assert trace["metadata"]["country"] == "RO"
        assert trace["metadata"]["network_type"] == "mobile_data"
        assert trace["metadata"]["bpf_filter"] == "tcp"
        assert trace["metadata"]["timestamp"] > 0
        assert len(trace["metadata"]["browser_trace"]) > 0


@pytest.mark.parametrize("use_dns_cache", [True, False])
def test_page_load_in_firefox_fail_dns_cache(
    use_dns_cache: bool,
) -> None:
    PAGE = "https://this.domain.does.not.exist/"
    traces = load_page_in_firefox(
        page_name=PAGE,
        use_dns_cache=use_dns_cache,
        http_caching_repeats=1,
        startup_wait_sec=1,
        page_load_timeout_sec=5,
        repeats=2,
        docker_name=f"selenium-firefox{OFFSET}",
        docker_iface=f"veth_firefox{OFFSET}",
        should_trace=False,
        bpf_filter="tcp",
        country="RO",
        network_type="mobile_data",
        remote_port=REMOTE_PORT,
    )
    assert len(traces) == 1
    for trace in traces:
        assert trace["page_body"] is None
        assert trace["stacktrace"] is not None
        assert trace["metadata"]["engine"] == f"selenium-firefox{OFFSET}"
        assert trace["metadata"]["use_dns_cache"] == use_dns_cache
        assert trace["metadata"]["http_caching_repeats"] == 1
        assert trace["metadata"]["country"] == "RO"
        assert trace["metadata"]["network_type"] == "mobile_data"
        assert trace["metadata"]["bpf_filter"] == "tcp"
        assert trace["metadata"]["timestamp"] > 0
        assert len(trace["metadata"]["browser_trace"]) == 0


@pytest.mark.parametrize(
    "user_agent",
    [
        "Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.92 Mobile Safari/537.36",
        "Mozilla/5.0 (Android 13; Mobile; rv:68.0) Gecko/68.0 Firefox/116.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
    ],
)
def test_page_load_in_firefox_user_agent(
    user_agent: str,
) -> None:
    PAGE = "https://www.google.com/"
    traces = load_page_in_firefox(
        page_name=PAGE,
        http_caching_repeats=1,
        startup_wait_sec=1,
        page_load_timeout_sec=5,
        repeats=2,
        docker_name=f"selenium-firefox{OFFSET}",
        docker_iface=f"veth_firefox{OFFSET}",
        should_trace=False,
        bpf_filter="tcp",
        country="RO",
        network_type="mobile_data",
        remote_port=REMOTE_PORT,
        # user_agent = user_agent,
    )
    assert len(traces) == 2
    for trace in traces:
        assert trace["page_body"] is not None
        assert trace["stacktrace"] is None
        # assert trace["metadata"]["user_agent"] == user_agent
        assert trace["metadata"]["http_caching_repeats"] == 1
        assert trace["metadata"]["country"] == "RO"
        assert trace["metadata"]["network_type"] == "mobile_data"
        assert trace["metadata"]["bpf_filter"] == "tcp"
        assert trace["metadata"]["timestamp"] > 0
        assert len(trace["metadata"]["browser_trace"]) > 0
