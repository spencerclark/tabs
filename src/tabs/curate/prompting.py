"""Shared prompt-construction helpers for the LLM curation passes.

Every prompt in this package that splices untrusted external content (feed titles,
summaries, fetched article bodies) into a request MUST build that block with
``wrap_untrusted`` rather than hand-rolling literal delimiter tags, so the structural
injection defense SPEC §6.5 calls for is implemented once and cannot drift between
call sites.
"""

import secrets


def wrap_untrusted(tag: str, body: str) -> str:
    """Wrap untrusted content in a delimiter an attacker embedded in `body` cannot forge.

    A per-call random suffix on the tag name means the closing tag the model is told to
    honor is not predictable from the prompt text alone, so untrusted content containing
    a literal "</tag>" (e.g. surfaced by HTML-entity unescaping upstream) cannot construct
    a matching close and break out of the block.
    """
    nonce = secrets.token_hex(8)
    opening = f"<{tag}_{nonce}>"
    closing = f"</{tag}_{nonce}>"
    return f"{opening}\n{body}\n{closing}"
