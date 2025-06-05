import requests
import time

def safe_request(url: str, max_retries: int = 3, backoff: float = 1.0, timeout: float = 10.0):
    """
    Perform an HTTP GET with retry logic.
    - url: the endpoint to GET.
    - max_retries: how many times to retry on failure.
    - backoff: multiplier (in seconds) between retries.
    - timeout: per-request timeout in seconds.

    Returns:
        The JSON-decoded response if status code is 200.
    Raises:
        RuntimeError if all retries fail or if status is not 200.
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                return response.json()
            else:
                raise RuntimeError(
                    f"HTTP {response.status_code} for {url}"
                )
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt < max_retries - 1:
                time.sleep(backoff * (2 ** attempt))
                continue
            else:
                raise RuntimeError(
                    f"Failed after {max_retries} retries for {url}"
                )
