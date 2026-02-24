"""Quick diagnostic: call vLLM for each failing task and print raw output."""
import sys
from mcn.config import MCNConfig

# Failing tasks
FAILING = [
    ("deduplicate", "def deduplicate(xs: list[int]) -> list[int]",
     "Remove duplicate elements from a list while preserving order."),
    ("fibonacci", "def fibonacci(n: int) -> int",
     "Compute the nth Fibonacci number (0-indexed: fib(0)=0, fib(1)=1)."),
    ("gcd", "def gcd(a: int, b: int) -> int",
     "Compute the greatest common divisor (GCD) of two positive integers."),
    ("reverse_string", "def reverse_string(s: str) -> str",
     "Reverse a string."),
    ("running_sum", "def running_sum(xs: list[int]) -> list[int]",
     "Compute the running sum of a list of integers."),
    ("partition", "def partition(xs: list[int], pivot: int) -> tuple[list[int], list[int]]",
     "Partition a list into two lists: elements less than pivot and elements greater or equal."),
    ("invert_dict", "def invert_dict(d: dict) -> dict",
     "Invert a dictionary: swap keys and values."),
]

SYSTEM = MCNConfig.TRIBE_PROMPTS[0]

from openai import OpenAI
client = OpenAI(base_url=MCNConfig.VLLM_BASE_URL, api_key=MCNConfig.VLLM_API_KEY)

from mcn.tribe import strip_code_fences

for func_name, sig, desc in FAILING:
    user_prompt = (
        f"Task: {desc}\n"
        f"Function name: {func_name}\n"
        f"Signature: {sig}\n"
        f"\nRespond with ONLY a ```python code block defining {func_name}. No explanation."
    )

    try:
        resp = client.chat.completions.create(
            model=MCNConfig.VLLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=512,
        )
        raw = resp.choices[0].message.content or ""
        code = strip_code_fences(raw)

        print(f"\n{'='*60}")
        print(f"TASK: {func_name}")
        print(f"{'='*60}")
        print(f"RAW RESPONSE ({len(raw)} chars):")
        print(raw[:500])
        print(f"\nEXTRACTED CODE:")
        print(code[:300])

        # Try to run it
        import ast
        try:
            ast.parse(code)
            print(f"\nSYNTAX: OK")
        except SyntaxError as e:
            print(f"\nSYNTAX ERROR: {e}")

        # Try to exec and test
        ns = {}
        try:
            exec(code, ns)
            if func_name in ns:
                print(f"FUNCTION FOUND: Yes")
            else:
                print(f"FUNCTION FOUND: No! Available: {[k for k in ns if not k.startswith('_')]}")
        except Exception as e:
            print(f"EXEC ERROR: {type(e).__name__}: {e}")

    except Exception as e:
        print(f"\n{func_name}: API ERROR - {e}")

print("\nDone.")
