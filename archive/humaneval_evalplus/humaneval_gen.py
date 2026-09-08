# EvalPlus codegen driver for a local finchmoe-infer (via shim.py).
#
# This is a near-verbatim copy of humaneval_3090/humaneval_gen.py (branch 3090,
# commit cc500e5). ONLY `root` and the optional id-range argument differ.
#
# DO NOT "clean up" the make_request monkeypatch below. On the 3090 it existed to
# dodge Windows' missing signal.SIGALRM, which macOS does not need — but the
# patch is ALSO what defines the protocol: upstream evalplus
# (evalplus/gen/util/openai_request.py) sends a user-only message list with NO
# system message. The system message and top_p=0.95 come from here. Changing
# this file breaks comparability with the published 3090 numbers.
#
# Note on max_tokens: the `max_tokens=512` default in make_request is dead code —
# evalplus's OpenAIChatDecoder always passes max_tokens=self.max_new_tokens
# explicitly, which defaults to 768 (evalplus/provider/base.py). The 3090 README's
# "max 512 new tokens" is therefore a documentation error; both sides really run
# at 768. Left as-is deliberately so both sides stay identical.
#
# Usage:
#   python humaneval_gen.py <model-id> [start end]
import os
import sys
import time

import openai

from evalplus.gen.util import openai_request
from evalplus import codegen

BASE = os.path.dirname(os.path.abspath(__file__))


def make_request(client, message, model, max_tokens=512, temperature=1.0, n=1, **kwargs):
    system_msg = "You are a helpful assistant good at coding."
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": message},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        n=n,
        top_p=0.95,
        timeout=600,
        **kwargs,
    )


def make_auto_request(*args, **kwargs):
    ret = None
    while ret is None:
        try:
            ret = make_request(*args, **kwargs)
        except openai.RateLimitError:
            print("Rate limit exceeded. Waiting...")
            time.sleep(5)
        except openai.APIConnectionError:
            print("API connection error. Waiting...")
            time.sleep(5)
        except openai.APIError as e:
            print(e)
            time.sleep(1)
        except Exception as e:
            print("Unknown error. Waiting...")
            print(e)
            time.sleep(1)
    return ret


openai_request.make_auto_request = make_auto_request

# Optional id range for smoke runs: `humaneval_gen.py finchmoe-3bit 0 20`
id_range = None
if len(sys.argv) >= 4:
    id_range = [int(sys.argv[2]), int(sys.argv[3])]

codegen.run_codegen(
    model=sys.argv[1] if len(sys.argv) > 1 else "finchmoe-3bit",
    dataset="humaneval",
    root=os.path.join(BASE, "results"),
    backend="openai",
    base_url="http://127.0.0.1:8080/v1",
    greedy=True,
    id_range=id_range,
)
