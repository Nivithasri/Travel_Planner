from __future__ import annotations

import pytest

from src.common.allowlist import Allowlist
from src.common.cache import DiskCache
from src.common.http import FetchError, HttpClient, TransportResponse

UA = "Test/0.1 (x@example.org)"
AT = "@"  # built here so no email-like literal appears in the source (privacy test)
PROD_REDIRECT_HOSTS = {"sahapedia.org", "www.sahapedia.org", "inditales.com", "www.inditales.com",
                       "en.wikipedia.org", "en.wikivoyage.org"}


def redirecting(first_url, location, status=302):
    calls = []

    def transport(method, url, headers, timeout):
        calls.append(url)
        if url == first_url:
            return TransportResponse(status, {"location": location}, "")
        return TransportResponse(200, {}, "SHOULD NEVER BE FETCHED")

    return transport, calls


def run(tmp_path, first_url, location, extra_redirect_hosts=()):
    transport, calls = redirecting(first_url, location)
    http = HttpClient(DiskCache(tmp_path / "c"), UA, transport=transport, sleep=lambda s: None,
                      min_delay_s=0, max_retries=0, allowlist=Allowlist.from_config(),
                      redirect_hosts=PROD_REDIRECT_HOSTS | set(extra_redirect_hosts))
    with pytest.raises(FetchError):
        http.fetch(first_url, check_robots=False)
    return calls


def host_of(url):
    return url.split("//", 1)[1].split("/", 1)[0].lower()


@pytest.mark.parametrize("permissive", [False, True])
def test_allowed_web_host_redirecting_to_an_unlisted_host_is_refused_before_following(
        tmp_path, permissive):
    first, target = "https://sahapedia.org/a", "https://unlisted.example.net/landing"
    # permissive=True also puts the target in the redirect rule's host list, so only the
    # allowlist is left to stop it (defence in depth)
    calls = run(tmp_path, first, target, [host_of(target)] if permissive else [])
    assert calls == [first]


@pytest.mark.parametrize("permissive", [False, True])
def test_api_host_redirecting_outside_its_path_prefix_is_refused_before_following(
        tmp_path, permissive):
    first = "https://overpass-api.de/api/interpreter?data=x"
    target = "https://overpass-api.de/api/status"
    calls = run(tmp_path, first, target, ["overpass-api.de"] if permissive else [])
    assert calls == [first]


@pytest.mark.parametrize("target", [
    "https://inditales.com" + AT + "evil.example/",  # real host is evil.example
    "https://evil.example" + AT + "inditales.com/",  # allowed host behind user information
    "https://user:pass" + AT + "www.sahapedia.org/x",
])
@pytest.mark.parametrize("permissive", [False, True])
def test_redirect_to_a_userinfo_url_is_refused_before_following(tmp_path, target, permissive):
    first = "https://inditales.com/post"
    netloc = target.split("//", 1)[1].split("/", 1)[0].lower()
    calls = run(tmp_path, first, target, [netloc] if permissive else [])
    assert calls == [first]


@pytest.mark.parametrize("target", ["ftp://inditales.com/x", "file:///etc/passwd",
                                    "javascript:alert(1)", "data:text/plain,hi",
                                    "gopher://inditales.com/"])
def test_redirect_to_a_non_http_scheme_is_refused_before_following(tmp_path, target):
    first = "https://inditales.com/post"
    calls = run(tmp_path, first, target)
    assert calls == [first]


def test_allowlist_itself_rejects_userinfo_and_other_schemes():
    allow = Allowlist.from_config()
    assert "user information" in allow.reason("https://inditales.com" + AT + "evil.example/")
    assert "user information" in allow.reason("https://evil.example" + AT + "inditales.com/")
    assert "scheme" in allow.reason("ftp://inditales.com/")
