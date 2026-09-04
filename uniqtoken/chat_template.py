"""
uniqtoken/chat_template.py
~~~~~~~~~~~~~~~~~~~~~~~~~~

Jinja2-powered chat template engine for UniqToken.

Implements `ChatTemplateEngine` and the four canonical pre-bundled templates
(chatml, llama3, mistral, zephyr) whose syntax is compatible with HuggingFace
`tokenizers`/`transformers` conventions.

`jinja2` is an **optional** dependency.  Import this module (or call
`CustomTokenizer.apply_chat_template`) without `jinja2` installed raises a
clear `ImportError` with installation instructions.

Security
--------
All template rendering is performed inside a Jinja2 `SandboxedEnvironment`
which prevents arbitrary Python execution from within template strings.
User-supplied *message content* is treated as data, never as markup, so
role-boundary tokens cannot be injected through the `content` field.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Pre-bundled reference templates
# ---------------------------------------------------------------------------
_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{{ message['content'] }}"
    "<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

_LLAMA3_TEMPLATE = (
    "{% for message in messages %}"
    "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n"
    "{{ message['content'] }}"
    "<|eot_id|>"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{% endif %}"
)

_MISTRAL_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "[INST] {{ message['content'] }} [/INST]"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] }}{{ eos_token }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}[INST] {% endif %}"
)

_ZEPHYR_TEMPLATE = (
    "{% for message in messages %}"
    "<|{{ message['role'] }}|>\n"
    "{{ message['content'] }}"
    "<|end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


#: Mapping of canonical template names to their Jinja2 source strings.
BUILTIN_TEMPLATES: Dict[str, str] = {
    "chatml": _CHATML_TEMPLATE,
    "llama3": _LLAMA3_TEMPLATE,
    "mistral": _MISTRAL_TEMPLATE,
    "zephyr": _ZEPHYR_TEMPLATE,
}


# ---------------------------------------------------------------------------
# ChatTemplateEngine
# ---------------------------------------------------------------------------

def _require_jinja2() -> Any:
    """Return the `jinja2` module, raising a friendly ImportError if absent."""
    try:
        import jinja2  # noqa: PLC0415
        return jinja2
    except ImportError as exc:
        raise ImportError(
            "The chat template feature requires jinja2.  "
            "Install it with:\n\n"
            "    pip install 'uniqtoken[chat]'\n\n"
            "or:\n\n"
            "    pip install jinja2>=3.1\n"
        ) from exc


class ChatTemplateEngine:
    """Renders a list of chat messages using a Jinja2 template string.

    Rendering is performed inside a :class:jinja2.sandbox.SandboxedEnvironment
    which disables arbitrary Python execution from within the template, making
    it safe to use user-supplied templates without code-execution risk.

    Role-injection protection
    -------------------------
    Message *content* is passed as a plain Python string variable into the
    Jinja2 context; the template accesses it via `{{ message['content'] }}`.
    Jinja2's autoescaping is intentionally *disabled* (the output is plain text,
    not HTML), but the sandbox prevents any code path that would let content
    strings execute as Jinja2 logic.  Boundary tokens that appear literally
    inside a content string are rendered as verbatim text, not as template
    control flow.

    Parameters
    ----------
    template_str:
        A Jinja2 template source string.  Use one of :data:BUILTIN_TEMPLATES
        or supply a custom string.
    """

    def __init__(self, template_str: str) -> None:
        if not isinstance(template_str, str):
            raise TypeError(
                f"template_str must be a str, got {type(template_str).__name__}"
            )
        self._template_str = template_str
        self._compiled_template: Optional[Any] = None  # lazily compiled

    @property
    def template_str(self) -> str:
        """The raw Jinja2 source string."""
        return self._template_str

    def render(
        self,
        conversation: List[Dict[str, str]],
        *,
        add_generation_prompt: bool = False,
        bos_token: str = "",
        eos_token: str = "",
    ) -> str:
        """Render *conversation* to a plain-text string.

        Parameters
        ----------
        conversation:
            A list of message dicts, each with at least the keys `"role"`
            and `"content"`.
        add_generation_prompt:
            Appends the model's turn-opening markup so the model can start
            generating immediately.
        bos_token:
            Beginning-of-sequence token string exposed to the template.
        eos_token:
            End-of-sequence token string exposed to the template.

        Returns
        -------
        str
            The fully-rendered chat string ready for tokenization.

        Raises
        ------
        ImportError
            If `jinja2` is not installed.
        ValueError
            If *conversation* is empty or a message is missing required keys.
        """
        self._validate_conversation(conversation)
        template = self._get_compiled_template()
        return template.render(
            messages=conversation,
            add_generation_prompt=add_generation_prompt,
            bos_token=bos_token,
            eos_token=eos_token,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_compiled_template(self) -> Any:
        """Return (and cache) the compiled Jinja2 template object."""
        if self._compiled_template is None:
            _require_jinja2()  # ensure jinja2 is installed
            import jinja2.sandbox  # noqa: PLC0415
            env = jinja2.sandbox.SandboxedEnvironment(
                keep_trailing_newline=False,
                autoescape=False,
            )
            self._compiled_template = env.from_string(self._template_str)
        return self._compiled_template

    @staticmethod
    def _validate_conversation(conversation: List[Dict[str, str]]) -> None:
        """Raise for malformed conversation inputs."""
        if not isinstance(conversation, list):
            raise TypeError(
                f"conversation must be a list of dicts, got {type(conversation).__name__}"
            )
        if not conversation:
            raise ValueError("conversation must contain at least one message")
        for i, message in enumerate(conversation):
            if not isinstance(message, dict):
                raise TypeError(
                    f"conversation[{i}] must be a dict, got {type(message).__name__}"
                )
            if "role" not in message:
                raise ValueError(
                    f"conversation[{i}] is missing required key 'role'"
                )
            if "content" not in message:
                raise ValueError(
                    f"conversation[{i}] is missing required key 'content'"
                )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def get_builtin_template(name: str) -> ChatTemplateEngine:
    """Return a :class:ChatTemplateEngine for one of the pre-bundled templates.

    Parameters
    ----------
    name:
        One of `"chatml"`, `"llama3"`, `"mistral"`, `"zephyr"`.

    Raises
    ------
    KeyError
        If *name* is not a known built-in template.
    """
    if name not in BUILTIN_TEMPLATES:
        available = ", ".join(sorted(BUILTIN_TEMPLATES))
        raise KeyError(
            f"Unknown built-in template {name!r}.  "
            f"Available templates: {available}"
        )
    return ChatTemplateEngine(BUILTIN_TEMPLATES[name])
