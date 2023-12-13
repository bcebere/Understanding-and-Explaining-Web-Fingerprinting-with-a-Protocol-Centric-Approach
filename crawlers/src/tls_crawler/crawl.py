# stdlib
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

# third party
import requests
from scapy.sendrecv import AsyncSniffer
from scapy.utils import wrpcap
from selenium import webdriver

# tls_crawler absolute
import tls_crawler.logger as log

MODULE_DIR = Path(__file__).parent


def get_docker_id(name: str) -> str:
    docker_id = (
        subprocess.check_output(["docker", "ps", "-aqf", f"name={name}"])
        .decode("utf-8")
        .split("\n")[0]
    )
    log.debug(f" Working with docker id {docker_id}")

    return docker_id


def refresh_dns_cache(docker_id: str) -> None:
    log.debug(f"Refresh DNS cache on {docker_id}")
    cache_reset = subprocess.check_output(
        [
            "docker",
            "exec",
            "-i",
            docker_id,
            "/bin/bash",
            "/home/seluser/dns_cache/refresh_unbound_cache.sh",
        ]
    )
    _ = subprocess.check_output(
        [
            "docker",
            "exec",
            "-i",
            docker_id,
            "sudo",
            "service",
            "unbound",
            "restart",
        ]
    )
    log.debug(f"DNS cache reset : {str(cache_reset)}")


def get_http_response_headers(url: str) -> Dict:
    req_worked = False
    for retry in range(3):
        try:
            response = requests.get(url, timeout=1)
        except BaseException:
            time.sleep(0.1)
            continue
        req_worked = True
        break

    if not req_worked:
        return {}

    headers = response.headers
    status_code = response.status_code

    interesting_headers = {
        "has_max_age": False,
        "has_max_age_null": False,
        "has_etag": False,
        "has_no_store": False,
        "has_no_cache": False,
        "has_must_revalidate": False,
        "status_code": status_code,
    }

    for header in headers:
        if "max-age" in headers[header]:
            interesting_headers["has_max_age"] = True
        if "max-age=0" in headers[header]:
            interesting_headers["has_max_age_null"] = True
        if "no-cache" in headers[header]:
            interesting_headers["has_no_cache"] = True
        if "no-store" in headers[header]:
            interesting_headers["has_no_store"] = True
        if "must-revalidate" in headers[header]:
            interesting_headers["has_must_revalidate"] = True
        if "etag" in header.lower():
            interesting_headers["has_etag"] = True

    return interesting_headers


def get_chrome_options(
    use_adblocking: bool,
    use_tracking_blocking: bool,
    use_http_caching: bool,
    use_images: bool = True,
    data_dir: Optional[Path] = None,
    user_agent: Optional[str] = None,
) -> webdriver.ChromeOptions:
    log.debug(
        f"""Generate chrome options with
                use_adblocking = {use_adblocking}
                use_tracking_blocking = {use_tracking_blocking}
                use_http_caching = {use_http_caching}
                use_images = {use_images}
                data_dir = {data_dir}
        """
    )
    extra_arguments = [
        "--disable-dev-shm-usage",
        "--ignore-certificate-errors",
        "--headless=new",
        "--disable-gpu",
        "--disable-notifications",
        "--disable-remote-fonts",
        "--disable-sync",
        "--window-size=1366,768",
        "--hide-scrollbars",
        "--disable-audio-output",
        "--dns-prefetch-disable",
        "--no-default-browser-check",
        "--disable-background-networking",
        "--enable-features=NetworkService,NetworkServiceInProcess",
        "--disable-background-timer-throttling",
        "--disable-backgrounding-occluded-windows",
        "--disable-breakpad",
        "--disable-client-side-phishing-detection",
        "--disable-component-extensions-with-background-pages",
        "--disable-default-apps",
        "--disable-features=TranslateUI",
        "--disable-hang-monitor",
        "--disable-ipc-flooding-protection",
        "--disable-prompt-on-repost",
        "--disable-renderer-backgrounding",
        "--force-color-profile=srgb",
        "--metrics-recording-only",
        "--no-first-run",
        "--password-store=basic",
        "--use-mock-keychain",
        "--disable-blink-features=AutomationControlled",
    ]
    options = webdriver.ChromeOptions()
    if data_dir is not None:
        options.add_argument(f"user-data-dir={data_dir}")
        options.add_argument(f"disk-cache-dir={data_dir}_cache")
    if user_agent is not None:
        options.add_argument(f"user-agent={user_agent}")

    if not use_http_caching:
        extra_arguments.extend(
            [
                "--disk-cache-size=0",
                "--aggressive-cache-discard",
            ]
        )
    if not use_images:
        extra_arguments.append("--blink-settings=imagesEnabled=false")

    options.set_capability("goog:loggingPrefs", {"browser": "ALL"})
    if use_adblocking:
        options.add_extension(
            MODULE_DIR / "extensions" / "chrome" / "AdBlock-—-best-ad-blocker.crx"
        )
    if use_tracking_blocking:
        options.add_extension(
            MODULE_DIR / "extensions" / "chrome" / "uBlock-Origin.crx"
        )

    for opt in extra_arguments:
        options.add_argument(opt)

    options.add_experimental_option("excludeSwitches", ["enable-automation"])

    return options


def post_chrome_setup(
    driver: webdriver.Remote,
    use_adblocking: bool,
    use_tracking_blocking: bool,
) -> webdriver.ChromeOptions:
    log.debug(
        f"""Chrome post setup callback
                use_adblocking = {use_adblocking}
                use_tracking_blocking = {use_tracking_blocking}
        """
    )


def get_firefox_options(
    use_adblocking: bool,
    use_tracking_blocking: bool,
    use_http_caching: bool,
    use_images: bool = True,
    data_dir: Optional[Path] = None,
    user_agent: Optional[str] = None,
) -> webdriver.FirefoxOptions:
    log.debug(
        f"""Generate Firefox options with
                use_adblocking = {use_adblocking}
                use_tracking_blocking = {use_tracking_blocking}
                use_http_caching = {use_http_caching}
                use_images = {use_images}
                data_dir = {data_dir}
        """
    )

    options = webdriver.FirefoxOptions()

    options.add_argument("--headless")
    if data_dir is not None:
        options.set_preference("browser.cache.disk.parent_directory", str(data_dir))

    options.set_preference("security.mixed_content.use_hstsc", False)
    options.set_preference("network.stricttransportsecurity.preloadlist", False)
    options.set_preference("extensions.getAddons.cache.enabled", False)
    options.set_preference("browser.cache.disk.enable", use_http_caching)
    options.set_preference("browser.cache.memory.enable", use_http_caching)
    options.set_preference("browser.cache.offline.enable", use_http_caching)
    options.set_preference("network.http.use-cache", use_http_caching)
    options.set_preference("toolkit.startup.max_resumed_crashes", "-1")
    options.set_preference("browser.startup.homepage_override.mstone", "ignore")
    options.set_preference("browser.search.geoip.url", "")
    options.set_preference("network.http.speculative-parallel-limit", 0)
    options.set_preference("network.dns.disablePrefetch", True)
    options.set_preference("network.prefetch-next", False)
    options.set_preference("OCSP.enabled", False)
    options.set_preference("security.ssl.enable_ocsp_stapling", False)
    options.set_preference("app.normandy.enabled", False)
    options.set_preference("browser.safebrowsing.downloads.remote.enabled", False)
    options.set_preference("extensions.blocklist.enabled", False)
    options.set_preference("network.captive-portal-service.enabled", False)
    options.set_preference("browser.aboutHomeSnippets.updateUrl", "")
    options.set_preference("devtools.chrome.enabled", True)
    options.set_preference("webdriver.load.strategy", "unstable")
    options.set_preference("privacy.trackingprotection.enabled", use_tracking_blocking)
    options.set_preference(
        "privacy.trackingprotection.annotate_channels", use_tracking_blocking
    )
    options.set_preference(
        "privacy.trackingprotection.cryptomining.enabled", use_tracking_blocking
    )
    options.set_preference(
        "privacy.trackingprotection.fingerprinting.enabled", use_tracking_blocking
    )
    if user_agent is not None:
        options.set_preference("general.useragent.override", user_agent)

    if not use_images:
        options.set_preference("permissions.default.image", 2)

    return options


def post_firefox_setup(
    driver: webdriver.Remote,
    use_adblocking: bool,
    use_tracking_blocking: bool,
) -> webdriver.FirefoxOptions:
    log.debug(
        f"""Post Firefox setup with
                use_adblocking = {use_adblocking}
                use_tracking_blocking = {use_tracking_blocking}
        """
    )
    if use_adblocking:
        webdriver.Firefox.install_addon(
            driver, MODULE_DIR / "extensions" / "firefox" / "adblocker_ultimate.xpi"
        )
    if use_tracking_blocking:
        webdriver.Firefox.install_addon(
            driver, MODULE_DIR / "extensions" / "firefox" / "ublock_origin.xpi"
        )


def hash_metadata(metadata: Dict) -> str:
    return hashlib.sha1(
        json.dumps(metadata, sort_keys=True).encode("utf-8")
    ).hexdigest()


def save_failure(workspace: Path, metadata: Dict, stacktrace: str, repeat: int) -> None:
    domain = metadata["request"]["netloc"].replace(".", "_")
    metadata_hash = hash_metadata(metadata)
    log.info(f"Saving failure {metadata_hash}")
    with open(workspace / f"{domain}_{metadata_hash}_{repeat}.meta", "w") as outfile:
        outfile.write(json.dumps(metadata, indent=4))
    with open(workspace / f"{domain}_{metadata_hash}_{repeat}.nx", "w") as outfile:
        outfile.write(stacktrace)


def save_trace(
    workspace: Path,
    metadata: Dict,
    pcap_trace: Any,
    browser_trace: List[Dict],
    repeat: int,
) -> None:
    domain = metadata["request"]["netloc"].replace(".", "_")
    metadata_hash = hash_metadata(metadata)
    metadata["browser_trace"] = browser_trace
    metadata["timestamp"] = time.time()

    log.info(f"Saving trace {metadata_hash}")
    with open(workspace / f"{domain}_{metadata_hash}_{repeat}.meta", "w") as outfile:
        outfile.write(json.dumps(metadata, indent=4))
    with open(workspace / f"{domain}_{metadata_hash}_{repeat}.pcap", "wb") as outfile:
        wrpcap(outfile, pcap_trace)


def load_page(
    page_name: str,
    docker_name: str,
    docker_iface: str,
    options_cbk: Callable,
    post_setup_cbk: Callable,
    use_adblocking: bool = False,
    use_tracking_blocking: bool = False,
    use_user_cookies: bool = False,
    use_http_caching: bool = False,
    use_dns_cache: bool = False,
    use_images: bool = True,
    http_caching_repeats: int = 5,
    remote_port: int = 4444,
    startup_wait_sec: int = 5,
    page_load_timeout_sec: int = 15,
    country: str = "DE",
    network_type: str = "wifi",
    repeats: int = 10,
    bpf_filter: str = "udp port 443 or tcp port 443 or udp port 53 or tcp port 53",
    should_trace: bool = True,
    should_save_trace: bool = True,
    workspace: Path = Path("network_traces"),
    user_agent: Optional[str] = None,
    group_id: Optional[Any] = None,
) -> List[Dict]:
    """
    Load a page in a Selenium backend and return the metadata, page body, or pcap trace.

    Arguments:
        page_name: str,
            Which page to load. Must include the protocol.
        docker_name: str
            Docker image to connect to.
        docker_iface: str
            Docker image to monitor for PCAP traces.
        options_cbk: Callable
            Driver callback for Options construction.
        post_setup_cbk: Callable
            Driver callback after the construction.
        use_adblocking: bool = False
            Enable/disable ad-blocking.
        use_tracking_blocking: bool = False
            Enable/disable tracking blocking.
        use_user_cookies: bool = False
            Enable/disable user cookies usage.
        use_http_caching: bool = False
            Enable/disable HTTP caching
        use_dns_cache: bool = False
            Enable/disable DNS caching
        use_images: bool = True
            Enable/disable image loading(smaller traces)
        http_caching_repeats: int = 5
            How many iterations to do for HTTP cache warming up.
        remote_port: int = 4444
            Selenium instrumentation port
        startup_wait_sec: int = 5
            Time to wait for extensions to load.
        page_load_timeout_sec: int = 15
            Time to wait for the page to load.
        country: str = "DE"
            Country label, used for metadata.
        network_type: str = "wifi"
            Source network, used for metadata.
        repeats: int = 10
            How many repeats of the same request to do.
        should_trace: bool = True,
            Enable network capture on the ``docker_iface``. This option might require sudo permissions.
        bpf_filter: str = "udp port 443 or tcp port 443 or udp port 53 or tcp port 53"
            When should_trace=True, which BPF filter to apply for the network capture.
        should_save_trace: bool = True,
            Enable network trace saving to PCAP.
        workspace: Path = Path("network_traces"),
            Folder for saving network traces
        user_agent: optional str
            Optional user agent
    """
    workspace.mkdir(parents=True, exist_ok=True)

    results = []
    page_tokens = urlparse(page_name)

    server_headers = get_http_response_headers(page_name)

    failed = False
    for repeat in range(repeats):
        metadata = {
            "request": {
                "scheme": page_tokens.scheme,
                "netloc": page_tokens.netloc,
                "path": page_tokens.path,
                "params": page_tokens.params,
                "query": page_tokens.query,
                "fragment": page_tokens.fragment,
                "original": page_name,
            },
            "use_adblocking": use_adblocking,
            "use_tracking_blocking": use_tracking_blocking,
            "use_user_cookies": use_user_cookies,
            "use_http_caching": use_http_caching,
            "use_dns_cache": use_dns_cache,
            "use_images": use_images,
            "http_caching_repeats": http_caching_repeats,
            "country": country,
            "network_type": network_type,
            "bpf_filter": bpf_filter,
            "first_server_headers": server_headers,
        }
        if user_agent is not None:
            metadata["user_agent"] = user_agent
        if group_id is not None:
            metadata["group_id"] = group_id

        repeat_result = {
            "page_body": "",
            "stacktrace": "",
            "metadata": metadata,
        }
        metadata_hash = hash_metadata(metadata)
        domain = page_tokens.netloc.replace(".", "_")

        if should_trace and should_save_trace:
            if (workspace / f"{domain}_{metadata_hash}_{repeat}.pcap").exists():
                log.info(f"Ignoring existing cache {domain} {repeat}")
                continue
            if (workspace / f"{domain}_{metadata_hash}_{repeat}.nx").exists():
                log.info(f"Ignoring previous failure {domain} {repeat}")
                continue

        log.info(f"Loading page {docker_iface}:  {metadata}")

        data_dir: Optional[Path] = None

        # Simulate HTTP caching
        if use_http_caching:
            # We create a dummy driver to preload the page and reuse the same data dir
            # We need a separate driver to prevent TLS session reusing.
            data_dir = Path("/tmp") / f"user_profile.{metadata_hash}"

            for retry in range(http_caching_repeats):
                log.debug(f"Warming the HTTP cache using {data_dir} {retry}")
                try:
                    options = options_cbk(
                        use_adblocking=use_adblocking,
                        use_tracking_blocking=use_tracking_blocking,
                        use_http_caching=use_http_caching,
                        use_images=use_images,
                        data_dir=data_dir,
                        user_agent=user_agent,
                    )
                    driver_cache = webdriver.Remote(
                        f"http://127.0.0.1:{remote_port}/wd/hub", options=options
                    )
                    driver_cache.set_page_load_timeout(page_load_timeout_sec)
                    post_setup_cbk(
                        driver_cache,
                        use_adblocking=use_adblocking,
                        use_tracking_blocking=use_tracking_blocking,
                    )
                    driver_cache.get(page_name)
                except BaseException as e:
                    log.error(f"Cache warming failed {page_name} : {e}")
                finally:
                    try:
                        driver_cache.quit()
                    except BaseException as e:
                        log.debug(f"driver.quit failed with error {e}")

        # Check DNS cache
        if not use_dns_cache:
            refresh_dns_cache(docker_name)

        # Configure Selenium
        log.debug("Create browser options")
        options = options_cbk(
            use_adblocking=use_adblocking,
            use_tracking_blocking=use_tracking_blocking,
            use_http_caching=use_http_caching,
            use_images=use_images,
            data_dir=data_dir,
            user_agent=user_agent,
        )

        try:
            log.debug("Create selenium driver")
            driver = webdriver.Remote(
                f"http://127.0.0.1:{remote_port}/wd/hub", options=options
            )
            driver.set_page_load_timeout(page_load_timeout_sec)

            log.debug("Run post-setup cbk")

            post_setup_cbk(
                driver,
                use_adblocking=use_adblocking,
                use_tracking_blocking=use_tracking_blocking,
            )

            # need to wait for the extension to load
            if use_adblocking or use_tracking_blocking:
                time.sleep(startup_wait_sec)

            if not use_user_cookies:
                driver.delete_all_cookies()

            if should_trace:
                log.debug("Create async sniffer")
                tracer = AsyncSniffer(
                    iface=docker_iface,
                    filter=bpf_filter,
                )

                log.debug("Start async sniffer")
                tracer.start()

                for retry in range(10):
                    if not hasattr(tracer, "stop_cb"):
                        log.debug(f"Tracer not ready yet {retry}")
                        time.sleep(startup_wait_sec)
                    else:
                        break

            log.debug(f"Load page {page_name}")
            driver.get(page_name)

            user_agent_browser = driver.execute_script("return navigator.userAgent")

            log.debug(
                f"Page body {page_name} user_agent={user_agent_browser} page_len =  {len(driver.page_source)}"
            )

            browser_trace = driver.execute_script(
                "return window.performance.getEntries();"
            )

            if should_trace:
                log.debug("Stop async sniffer")
                network_trace = tracer.stop()

                if should_save_trace:
                    save_trace(
                        workspace, metadata, network_trace, browser_trace, repeat
                    )

                del network_trace
                del tracer

            metadata["browser_trace"] = browser_trace
            metadata["timestamp"] = time.time()
            repeat_result["metadata"] = metadata
            repeat_result["page_body"] = driver.page_source
            repeat_result["stacktrace"] = None  # type: ignore
        except BaseException as e:
            stacktrace = traceback.format_exc()
            if should_save_trace:
                save_failure(workspace, metadata, stacktrace, repeat)

            metadata["browser_trace"] = []
            metadata["timestamp"] = time.time()
            repeat_result["metadata"] = metadata
            repeat_result["page_body"] = None  # type: ignore
            repeat_result["stacktrace"] = stacktrace

            log.error(stacktrace)
            log.error(f"Request failed {page_name} : {e}")
            failed = True
        finally:
            try:
                driver.quit()
            except BaseException as e:
                log.debug(f"driver.quit failed with error {e}")
            gc.collect()  # Free the memory from Scapy

        results.append(repeat_result)
        if failed:
            break

    return results


def load_page_in_chrome(
    page_name: str,
    docker_name: str = "selenium-chrome",
    docker_iface: str = "veth_chrome",
    use_adblocking: bool = False,
    use_tracking_blocking: bool = False,
    use_user_cookies: bool = False,
    use_http_caching: bool = False,
    use_dns_cache: bool = False,
    use_images: bool = True,
    http_caching_repeats: int = 5,
    remote_port: int = 4444,
    startup_wait_sec: int = 5,
    country: str = "DE",
    network_type: str = "wifi",
    repeats: int = 10,
    page_load_timeout_sec: int = 15,
    bpf_filter: str = "udp port 443 or tcp port 443 or udp port 53 or tcp port 53",
    should_trace: bool = True,
    should_save_trace: bool = True,
    workspace: Path = Path("network_traces"),
    user_agent: Optional[str] = None,
    group_id: Optional[Any] = None,
) -> List[Dict]:
    """
    Load a page in a Selenium ChromeDriver and return the metadata, page body, or pcap trace.

    Arguments:
        page_name: str,
            Which page to load. Must include the protocol.
        docker_name: str
            Docker image to connect to.
        docker_iface: str
            Docker image to monitor for PCAP traces.
        options_cbk: Callable
            Driver callback for Options construction.
        post_setup_cbk: Callable
            Driver callback after the construction.
        use_adblocking: bool = False
            Enable/disable ad-blocking.
        use_tracking_blocking: bool = False
            Enable/disable tracking blocking.
        use_user_cookies: bool = False
            Enable/disable user cookies usage.
        use_http_caching: bool = False
            Enable/disable HTTP caching
        use_dns_cache: bool = False
            Enable/disable DNS caching
        use_images: bool = True
            Enable/disable image loading(smaller traces)
        http_caching_repeats: int = 5
            How many iterations to do for HTTP cache warming up.
        remote_port: int = 4444
            Selenium instrumentation port
        startup_wait_sec: int = 5
            Time to wait for extensions to load.
        page_load_timeout_sec: int = 15
            Time to wait for the page to load.
        country: str = "DE"
            Country label, used for metadata.
        network_type: str = "wifi"
            Source network, used for metadata.
        repeats: int = 10
            How many repeats of the same request to do.
        should_trace: bool = True,
            Enable network capture on the ``docker_iface``. This option might require sudo permissions.
        bpf_filter: str = "udp port 443 or tcp port 443 or udp port 53 or tcp port 53"
            When should_trace=True, which BPF filter to apply for the network capture.
        should_save_trace: bool = True,
            Enable network trace saving to PCAP.
        workspace: Path = Path("network_traces"),
            Folder for saving network traces
        user_agent: Optional str
            Optional custom user agent
    """
    return load_page(
        page_name=page_name,
        docker_name=docker_name,
        docker_iface=docker_iface,
        options_cbk=get_chrome_options,
        post_setup_cbk=post_chrome_setup,
        use_adblocking=use_adblocking,
        use_tracking_blocking=use_tracking_blocking,
        use_user_cookies=use_user_cookies,
        use_http_caching=use_http_caching,
        use_dns_cache=use_dns_cache,
        use_images=use_images,
        http_caching_repeats=http_caching_repeats,
        remote_port=remote_port,
        startup_wait_sec=startup_wait_sec,
        country=country,
        network_type=network_type,
        repeats=repeats,
        page_load_timeout_sec=page_load_timeout_sec,
        bpf_filter=bpf_filter,
        should_trace=should_trace,
        should_save_trace=should_save_trace,
        workspace=workspace,
        user_agent=user_agent,
        group_id=group_id,
    )


def load_page_in_firefox(
    page_name: str,
    docker_name: str = "selenium-firefox",
    docker_iface: str = "veth_firefox",
    use_adblocking: bool = False,
    use_tracking_blocking: bool = False,
    use_user_cookies: bool = False,
    use_http_caching: bool = False,
    use_dns_cache: bool = False,
    use_images: bool = True,
    http_caching_repeats: int = 5,
    remote_port: int = 4445,
    startup_wait_sec: int = 5,
    country: str = "DE",
    network_type: str = "wifi",
    repeats: int = 10,
    page_load_timeout_sec: int = 15,
    bpf_filter: str = "udp port 443 or tcp port 443 or udp port 53 or tcp port 53",
    should_trace: bool = True,
    should_save_trace: bool = True,
    workspace: Path = Path("network_traces"),
    user_agent: Optional[str] = None,
    group_id: Optional[Any] = None,
) -> List[Dict]:
    """
    Load a page in a Selenium FirefoxDriver and return the metadata, page body, or pcap trace.

    Arguments:
        page_name: str,
            Which page to load. Must include the protocol.
        docker_name: str
            Docker image to connect to.
        docker_iface: str
            Docker image to monitor for PCAP traces.
        options_cbk: Callable
            Driver callback for Options construction.
        post_setup_cbk: Callable
            Driver callback after the construction.
        use_adblocking: bool = False
            Enable/disable ad-blocking.
        use_tracking_blocking: bool = False
            Enable/disable tracking blocking.
        use_user_cookies: bool = False
            Enable/disable user cookies usage.
        use_http_caching: bool = False
            Enable/disable HTTP caching
        use_dns_cache: bool = False
            Enable/disable DNS caching
        use_images: bool = True
            Enable/disable image loading(smaller traces)
        http_caching_repeats: int = 5
            How many iterations to do for HTTP cache warming up.
        remote_port: int = 4444
            Selenium instrumentation port
        startup_wait_sec: int = 5
            Time to wait for extensions to load.
        page_load_timeout_sec: int = 15
            Time to wait for the page to load.
        country: str = "DE"
            Country label, used for metadata.
        network_type: str = "wifi"
            Source network, used for metadata.
        repeats: int = 10
            How many repeats of the same request to do.
        should_trace: bool = True,
            Enable network capture on the ``docker_iface``. This option might require sudo permissions.
        bpf_filter: str = "udp port 443 or tcp port 443 or udp port 53 or tcp port 53"
            When should_trace=True, which BPF filter to apply for the network capture.
        should_save_trace: bool = True,
            Enable network trace saving to PCAP.
        workspace: Path = Path("network_traces"),
            Folder for saving network traces
        user_agent: Optional str
            optional user agent
    """
    return load_page(
        page_name=page_name,
        docker_name=docker_name,
        docker_iface=docker_iface,
        options_cbk=get_firefox_options,
        post_setup_cbk=post_firefox_setup,
        use_adblocking=use_adblocking,
        use_tracking_blocking=use_tracking_blocking,
        use_user_cookies=use_user_cookies,
        use_http_caching=use_http_caching,
        use_dns_cache=use_dns_cache,
        use_images=use_images,
        http_caching_repeats=http_caching_repeats,
        remote_port=remote_port,
        startup_wait_sec=startup_wait_sec,
        country=country,
        network_type=network_type,
        repeats=repeats,
        page_load_timeout_sec=page_load_timeout_sec,
        bpf_filter=bpf_filter,
        should_trace=should_trace,
        should_save_trace=should_save_trace,
        workspace=workspace,
        user_agent=user_agent,
        group_id=group_id,
    )


def load_pages_in_chrome(
    page_names: List[str],
    docker_name: str = "selenium-chrome",
    docker_iface: str = "veth_chrome",
    use_adblocking: bool = False,
    use_tracking_blocking: bool = False,
    use_user_cookies: bool = False,
    use_http_caching: bool = False,
    use_dns_cache: bool = False,
    use_images: bool = True,
    http_caching_repeats: int = 5,
    remote_port: int = 4444,
    startup_wait_sec: int = 5,
    country: str = "DE",
    network_type: str = "wifi",
    repeats: int = 10,
    page_load_timeout_sec: int = 15,
    bpf_filter: str = "udp port 443 or tcp port 443 or udp port 53 or tcp port 53",
    should_trace: bool = True,
    should_save_trace: bool = True,
    workspace: Path = Path("network_traces"),
    user_agent: Optional[str] = None,
) -> None:
    """
    Load mtuiple pages in a Selenium ChromeDriver and return the metadata, page body, or pcap trace.

    Arguments:
        page_name: str,
            Which page to load. Must include the protocol.
        docker_name: str
            Docker image to connect to.
        docker_iface: str
            Docker image to monitor for PCAP traces.
        options_cbk: Callable
            Driver callback for Options construction.
        post_setup_cbk: Callable
            Driver callback after the construction.
        use_adblocking: bool = False
            Enable/disable ad-blocking.
        use_tracking_blocking: bool = False
            Enable/disable tracking blocking.
        use_user_cookies: bool = False
            Enable/disable user cookies usage.
        use_http_caching: bool = False
            Enable/disable HTTP caching
        use_dns_cache: bool = False
            Enable/disable DNS caching
        use_images: bool = True
            Enable/disable image loading(smaller traces)
        http_caching_repeats: int = 5
            How many iterations to do for HTTP cache warming up.
        remote_port: int = 4444
            Selenium instrumentation port
        startup_wait_sec: int = 5
            Time to wait for extensions to load.
        page_load_timeout_sec: int = 15
            Time to wait for the page to load.
        country: str = "DE"
            Country label, used for metadata.
        network_type: str = "wifi"
            Source network, used for metadata.
        repeats: int = 10
            How many repeats of the same request to do.
        should_trace: bool = True,
            Enable network capture on the ``docker_iface``. This option might require sudo permissions.
        bpf_filter: str = "udp port 443 or tcp port 443 or udp port 53 or tcp port 53"
            When should_trace=True, which BPF filter to apply for the network capture.
        should_save_trace: bool = True,
            Enable network trace saving to PCAP.
        workspace: Path = Path("network_traces"),
            Folder for saving network traces
        user_agent: Optional str
            Optional custom user agent
    """

    def _load_page(page_name: str) -> Any:
        return load_page_in_chrome(
            page_name=page_name,
            docker_name=docker_name,
            docker_iface=docker_iface,
            use_adblocking=use_adblocking,
            use_tracking_blocking=use_tracking_blocking,
            use_user_cookies=use_user_cookies,
            use_http_caching=use_http_caching,
            use_dns_cache=use_dns_cache,
            use_images=use_images,
            http_caching_repeats=http_caching_repeats,
            remote_port=remote_port,
            startup_wait_sec=startup_wait_sec,
            country=country,
            network_type=network_type,
            repeats=repeats,
            page_load_timeout_sec=page_load_timeout_sec,
            bpf_filter=bpf_filter,
            should_trace=should_trace,
            should_save_trace=should_save_trace,
            workspace=workspace,
            user_agent=user_agent,
            group_id=page_names,
        )

    threads = []

    for url in page_names:
        thread = threading.Thread(target=_load_page, args=(url,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
