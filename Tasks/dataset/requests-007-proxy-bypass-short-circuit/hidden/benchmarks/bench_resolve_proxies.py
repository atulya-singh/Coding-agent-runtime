"""Performance criterion for requests-007. Prints one float to stdout.

resolve_proxies(trust_env=False) must not pay for should_bypass_proxies(), which
is O(len(no_proxy)). We time that call against the same call with trust_env=True
-- the path that legitimately still has to do the work -- and print the ratio.

Reporting a *ratio* rather than wall-clock seconds is deliberate: the control is
measured in the same process, on the same machine, under the same CPU quota, so
one threshold stays meaningful on a laptop, in CI, and inside a throttled
container. Each side takes the minimum of several repeats, which is the standard
way to read a timing distribution whose noise is all one-sided.

Measured on the unfixed base commit: ~0.50. With the fix: ~0.00003.
"""
import timeit

from requests.models import PreparedRequest
from requests.utils import resolve_proxies

# Large enough that the scan dominates, and deliberately containing no entry
# that matches the request host -- a match would short-circuit the scan and
# measure nothing.
NO_PROXY = ",".join(f"host{i}.example.internal" for i in range(20000))
PROXIES = {"no_proxy": NO_PROXY}

REQUEST = PreparedRequest()
REQUEST.prepare_url("http://target.example.com/path", None)

CALLS = 20
REPEATS = 5


def best(trust_env):
    timer = timeit.Timer(lambda: resolve_proxies(REQUEST, PROXIES, trust_env))
    return min(timer.repeat(repeat=REPEATS, number=CALLS))


trust_env_off = best(False)
trust_env_on = best(True)
print(f"{trust_env_off / trust_env_on:.6f}")
