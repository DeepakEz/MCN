"""MCN-v0.1 Tribe Agent — LLM-Based Code Generator.

Each Tribe is a Ray actor wrapping a code-generation LLM.
Ti = (pi_i, theta_i, B_i) in the formal spec.

In v0.1:
    - pi_i   = LLM completion call (OpenAI client -> vLLM)
    - theta_i = system prompt + few-shot patch context (no weight updates)
    - B_i    = Campfire buffer (stored here, consumed by council for patches)

The generate() method:
    1. Builds a prompt from task spec + system prompt + patches
    2. Calls OpenAI-compatible API (local vLLM or any OpenAI endpoint)
    3. Strips markdown code fences from response
    4. Returns (code, plan) tuple
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import ray

from mcn.config import MCNConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Markdown fence stripper
# ---------------------------------------------------------------------------

# Matches ```python ... ``` or ``` ... ``` blocks, capturing inner content
# Flexible: allows optional whitespace/newline after language tag
_CODE_FENCE_RE = re.compile(
    r"```(?:python|py)?[ \t]*\r?\n(.*?)```",
    re.DOTALL,
)


def strip_code_fences(text: str) -> str:
    """Extract code from markdown fenced blocks with syntax validation.

    If the response contains ```python ... ``` blocks, extract and
    concatenate their contents. If no fences found, try to extract
    bare function definitions. Always validates syntax.

    Args:
        text: Raw LLM response text.

    Returns:
        Cleaned Python source code.
    """
    import ast

    # Strategy 1: Extract from markdown code fences
    matches = _CODE_FENCE_RE.findall(text)
    if matches:
        code = "\n\n".join(m.strip() for m in matches)
        try:
            ast.parse(code)
            return code
        except SyntaxError:
            pass  # Fall through to other strategies

    # Strategy 2: Find bare function definition(s) in the text
    lines = text.split("\n")
    code_lines: list[str] = []
    capturing = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("def "):
            capturing = True
        if capturing:
            # Stop capturing on blank lines after top-level code,
            # or lines that look like prose (no indent, not def/class/return/if/etc)
            if (
                stripped
                and not stripped.startswith(("def ", "class ", " ", "\t", "#"))
                and not line.startswith((" ", "\t"))
                and code_lines
                and not stripped.startswith(("return", "if ", "else", "elif",
                                            "for ", "while ", "try", "except",
                                            "with ", "import ", "from ", "raise"))
            ):
                break
            code_lines.append(line)

    if code_lines:
        code = "\n".join(code_lines).strip()
        try:
            ast.parse(code)
            return code
        except SyntaxError:
            pass

    # Strategy 3: Return everything and hope for the best
    return text.strip()


# ---------------------------------------------------------------------------
# Generation result (internal message to council)
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """Output from a Tribe's generate() call."""
    tribe_id: str
    code: str = ""
    plan: str = ""
    raw_response: str = ""
    tokens_used: int = 0
    error: str = ""          # non-empty if generation failed


# ---------------------------------------------------------------------------
# Ray Actor
# ---------------------------------------------------------------------------

@ray.remote
class TribeActor:
    """LLM-based code generation agent.

    Each tribe has a distinct system prompt that shapes its coding style
    (e.g., 'Reliable Coder', 'Fast Coder', 'Creative Coder').

    Patches (few-shot exemplars from other tribes) are appended to the
    system prompt as context injections — this is the v0.1 diffusion
    mechanism: theta_j += lambda * P_ij becomes "append examples to prompt."

    Args:
        tribe_id: Unique identifier for this tribe.
        system_prompt: Base system prompt defining the tribe's personality.
        model: Model name (must match vLLM --model flag, or any OpenAI model).
        temperature: Sampling temperature.
        max_tokens: Max tokens for generation.
        use_mock: If True, return a deterministic mock response (for testing).
    """

    def __init__(
        self,
        tribe_id: str,
        system_prompt: str,
        model: str = "",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        use_mock: bool = False,
        failure_bias: str = "",
        vllm_base_url: str = "",
    ) -> None:
        self.tribe_id = tribe_id
        self.system_prompt = system_prompt
        self.model = model or MCNConfig.VLLM_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_mock = use_mock
        self.vllm_base_url = vllm_base_url or MCNConfig.VLLM_BASE_URL

        # failure_bias: when set (e.g. "memory", "timeout", "index"),
        # mock mode will inject a bug when the task description contains
        # this keyword. This simulates tribal specialization weaknesses
        # so the Council can learn to route around them.
        self.failure_bias = failure_bias.lower()

        # Patch context: appended to system prompt (few-shot exemplars)
        # This is the P_ij injection point — token-bounded
        self._patch_context: str = ""
        self._patch_token_budget: int = 1500  # L_max for patches

    # ------------------------------------------------------------------
    # Patch management (knowledge diffusion interface)
    # ------------------------------------------------------------------

    def apply_patch(self, patch_text: str) -> bool:
        """Inject a few-shot patch into this tribe's context.

        Returns False if the patch would exceed the token budget.
        Token count is approximated as len(text) // 4.
        """
        approx_tokens = len(patch_text) // 4
        current_tokens = len(self._patch_context) // 4

        if current_tokens + approx_tokens > self._patch_token_budget:
            logger.warning(
                "Tribe %s: patch rejected — would exceed token budget "
                "(%d + %d > %d)",
                self.tribe_id, current_tokens, approx_tokens,
                self._patch_token_budget,
            )
            return False

        self._patch_context += "\n\n" + patch_text
        logger.info(
            "Tribe %s: patch applied (%d tokens, total %d)",
            self.tribe_id, approx_tokens, current_tokens + approx_tokens,
        )
        return True

    def clear_patches(self) -> None:
        """Remove all injected patches (reset to base prompt)."""
        self._patch_context = ""

    def get_effective_prompt(self) -> str:
        """Return the full system prompt including patches (for debugging)."""
        if self._patch_context:
            return self.system_prompt + "\n\n" + self._patch_context
        return self.system_prompt

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------

    def generate(
        self,
        task_description: str,
        function_name: str,
        input_signature: str,
        unit_tests: str = "",
        hint: str = "",
        reference_solution: str | None = None,
    ) -> GenerationResult:
        """Generate Python code for a task.

        Args:
            task_description: Natural-language spec.
            function_name: Expected entry-point name.
            input_signature: Type signature string.
            unit_tests: Reference tests (shown to the tribe for context).
            hint: Optional hint from previous failure (Campfire reflection).
            reference_solution: Optional canonical implementation shown to the
                model as a "reference approach". Useful for tasks where small
                models consistently choose a wrong algorithm (e.g. naive
                recursion when iteration is required). The model is still free
                to differ, but the pattern acts as a strong in-context prior.

        Returns:
            GenerationResult with code, plan, and metadata.
        """
        result = GenerationResult(tribe_id=self.tribe_id)

        # Build the user prompt
        user_prompt = self._build_user_prompt(
            task_description, function_name, input_signature,
            unit_tests, hint, reference_solution,
        )

        if self.use_mock:
            return self._mock_generate(result, function_name)

        try:
            from openai import OpenAI

            client = OpenAI(
                base_url=self.vllm_base_url,
                api_key=MCNConfig.VLLM_API_KEY,
            )

            effective_system = self.get_effective_prompt()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": effective_system},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            raw = response.choices[0].message.content or ""
            result.raw_response = raw
            result.tokens_used = response.usage.total_tokens if response.usage else 0

            # Extract code and plan
            result.code = strip_code_fences(raw)
            result.plan = self._extract_plan(raw)

            # Syntax check — if code is broken, log it
            import ast
            try:
                ast.parse(result.code)
            except SyntaxError as syn_err:
                logger.warning(
                    "Tribe %s: extracted code has SyntaxError: %s",
                    self.tribe_id, syn_err,
                )
                result.error = f"SyntaxError in generated code: {syn_err}"

        except Exception as e:
            result.error = f"Generation failed: {type(e).__name__}: {e}"
            logger.error("Tribe %s generation error: %s", self.tribe_id, e)

        return result

    def _build_user_prompt(
        self,
        task_description: str,
        function_name: str,
        input_signature: str,
        unit_tests: str,
        hint: str,
        reference_solution: str | None = None,
    ) -> str:
        """Construct the user-facing prompt for code generation."""
        parts = [
            f"Task: {task_description}",
            f"Function name: {function_name}",
            f"Signature: {input_signature}",
        ]

        if reference_solution:
            parts.append(
                f"Reference approach (adapt this pattern):\n"
                f"```python\n{reference_solution}\n```"
            )

        if unit_tests:
            parts.append(
                f"Tests to pass:\n```python\n{unit_tests}\n```"
            )

        if hint:
            parts.append(f"Previous error: {hint}")

        parts.append(
            f"\nRespond with ONLY a ```python code block defining {function_name}. "
            f"No explanation."
        )

        return "\n".join(parts)

    def _extract_plan(self, raw_response: str) -> str:
        """Extract the planning/reasoning text before the code block."""
        # Everything before the first code fence is the "plan"
        fence_start = raw_response.find("```")
        if fence_start > 0:
            return raw_response[:fence_start].strip()
        return ""

    def _mock_generate(
        self, result: GenerationResult, function_name: str,
    ) -> GenerationResult:
        """Return a mock response for testing, with optional failure injection.

        If self.failure_bias is set and the task description (embedded in
        the user prompt built earlier) matches the bias keyword, generate
        BUGGY code that triggers a specific exception type. This simulates
        tribal specialization weaknesses so the Council can learn routing.

        Bias -> Bug mapping:
          "memory"  -> code that allocates a huge list (MemoryError)
          "timeout" -> code with an infinite loop (TimeoutError)
          "index"   -> code that accesses xs[999999] (IndexError)
          "type"    -> code that does int + str (TypeError)
          "key"     -> code that accesses dict["missing"] (KeyError)
          "recurse" -> code with infinite recursion (RecursionError)

        If no bias matches the task, generate correct code.
        """
        # Check if this tribe should inject a bug for this task
        inject_bug = self._should_inject_failure(function_name)

        if inject_bug:
            return self._generate_buggy_code(result, function_name)

        # Default: generate correct code
        result.code = (
            f"def {function_name}(xs):\n"
            f"    \"\"\"Process a list.\"\"\"\n"
            f"    if not isinstance(xs, list):\n"
            f"        raise TypeError('Expected a list')\n"
            f"    return sorted(xs)\n"
        )
        result.plan = (
            f"[{self.tribe_id}] Using Python's built-in sorted() "
            f"for a stable, O(n log n) sort."
        )
        result.raw_response = result.plan + "\n```python\n" + result.code + "\n```"
        result.tokens_used = 0
        return result

    def _should_inject_failure(self, function_name: str) -> bool:
        """Decide if this tribe should inject a bug for this task.

        The bias keyword is matched against the function_name.
        E.g., if failure_bias="memory" and function_name contains "memory",
        this tribe will produce buggy code for this task.
        """
        if not self.failure_bias:
            return False
        return self.failure_bias in function_name.lower()

    def _generate_buggy_code(
        self, result: GenerationResult, function_name: str,
    ) -> GenerationResult:
        """Generate code with a deliberate bug matching the failure_bias."""
        bias = self.failure_bias

        if bias == "memory":
            # Allocates a massive list -> MemoryError
            result.code = (
                f"def {function_name}(xs):\n"
                f"    big = [0] * (10 ** 10)\n"
                f"    return sorted(xs)\n"
            )
            result.plan = f"[{self.tribe_id}] BUG INJECTED: MemoryError"

        elif bias == "timeout":
            # Infinite loop -> TimeoutError
            result.code = (
                f"def {function_name}(xs):\n"
                f"    while True:\n"
                f"        pass\n"
                f"    return sorted(xs)\n"
            )
            result.plan = f"[{self.tribe_id}] BUG INJECTED: Timeout"

        elif bias == "index":
            # Out-of-range access -> IndexError
            result.code = (
                f"def {function_name}(xs):\n"
                f"    return xs[999999]\n"
            )
            result.plan = f"[{self.tribe_id}] BUG INJECTED: IndexError"

        elif bias == "type":
            # Type mismatch -> TypeError
            result.code = (
                f"def {function_name}(xs):\n"
                f"    return xs + 42\n"
            )
            result.plan = f"[{self.tribe_id}] BUG INJECTED: TypeError"

        elif bias == "key":
            # Missing dict key -> KeyError
            result.code = (
                f"def {function_name}(xs):\n"
                f"    d = {{}}\n"
                f"    return d['missing']\n"
            )
            result.plan = f"[{self.tribe_id}] BUG INJECTED: KeyError"

        elif bias == "recurse":
            # Infinite recursion -> RecursionError
            result.code = (
                f"def {function_name}(xs):\n"
                f"    return {function_name}(xs)\n"
            )
            result.plan = f"[{self.tribe_id}] BUG INJECTED: RecursionError"

        else:
            # Unknown bias: just generate correct code
            result.code = (
                f"def {function_name}(xs):\n"
                f"    return sorted(xs)\n"
            )
            result.plan = f"[{self.tribe_id}] No matching bias, correct code."

        result.raw_response = result.plan + "\n```python\n" + result.code + "\n```"
        result.tokens_used = 0
        return result

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def get_tribe_id(self) -> str:
        """Return this tribe's identifier."""
        return self.tribe_id

    def get_info(self) -> dict:
        """Return diagnostic info about this tribe."""
        return {
            "tribe_id": self.tribe_id,
            "model": self.model,
            "use_mock": self.use_mock,
            "patch_tokens": len(self._patch_context) // 4,
            "patch_budget": self._patch_token_budget,
        }
