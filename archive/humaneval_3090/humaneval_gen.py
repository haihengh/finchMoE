# Windows-safe evalplus codegen driver for a local llama-server.
# Replaces evalplus.gen.util.openai_request.make_auto_request, which uses
# signal.SIGALRM (not available on Windows), with a plain SDK-timeout version.
import sys
import time

import openai

from evalplus.gen.util import openai_request
from evalplus import codegen


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

codegen.run_codegen(
    model=sys.argv[1] if len(sys.argv) > 1 else "qwen3.8-27b",
    dataset="humaneval",
    root="D:/code/llama.cpp/evals/evalplus_results",
    backend="openai",
    base_url="http://127.0.0.1:8080/v1",
    greedy=True,
)
