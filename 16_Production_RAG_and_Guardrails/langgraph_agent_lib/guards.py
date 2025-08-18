#==============================================
# Guardrails
#==============================================

import warnings
warnings.filterwarnings(
    "ignore",
    message="Could not obtain an event loop",
    category=UserWarning,
    module="guardrails.validator_service"
)

# testing guardrails sync (no asyncio concurrency)

import inspect
from typing import Dict, Any
from pprint import pprint
import asyncio

def _dedupe(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def _call_guard_sync(name: str, guard, payload, metadata: dict | None = None) -> dict:
    """Run a single guard synchronously, classify outcome as passed/redacted/blocked."""
    try:
        # Most guards accept a single positional 'value'
        res = guard.validate(payload, metadata=metadata) if metadata is not None else guard.validate(payload)

        # If a guard returns an awaitable (rare), resolve it synchronously
        if inspect.isawaitable(res):
            # This blocks until the awaitable completes, but keeps this API synchronous.
            import asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                # Nested running loop (e.g., Jupyter): create a new one in a separate context
                # Use asyncio.run in a nested function to avoid interfering with the notebook loop
                async def _await_res(r): return await r
                res = asyncio.run(_await_res(res))
            else:
                async def _await_res(r): return await r
                res = asyncio.run(_await_res(res))

        # Interpret Guardrails' ValidationOutcome if present
        validation_passed = getattr(res, "validation_passed", None)

        action = "passed"
        reasons = []
        pii_types = []
        sanitized_text = getattr(res, "validated_output", None)
        raw_text = getattr(res, "raw_llm_output", None)

        summaries = getattr(res, "validation_summaries", []) or []
        failed_any = any(getattr(s, "validator_status", "").lower() == "fail" for s in summaries)

        # Collect reasons & PII entity types from spans
        for s in summaries:
            if getattr(s, "validator_status", "").lower() == "fail":
                fr = getattr(s, "failure_reason", None)
                if fr:
                    reasons.append(fr)
                for span in getattr(s, "error_spans", []) or []:
                    r = getattr(span, "reason", None)
                    if r:
                        reasons.append(r)
                        pii_types.append(r)

        # Redaction detection: output changed or there are failed summaries
        redacted = (
            (sanitized_text is not None and raw_text is not None and sanitized_text != raw_text)
            or failed_any
        )

        # Decide final action
        if isinstance(validation_passed, bool):
            if not validation_passed:
                action = "blocked"
            elif redacted:
                action = "redacted"
            else:
                action = "passed"
        else:
            action = "passed"  # Non-ValidationOutcome: treat as pass

        result = {
            "status": "success",
            "success": True,
            "guardrail_name": name,
            "validation_passed": bool(validation_passed) if isinstance(validation_passed, bool) else True,
            "filtered": (action == "blocked"),
            "action": action,                 # "passed" | "redacted" | "blocked"
            "reasons": _dedupe(reasons),
            "pii_types": _dedupe(pii_types),
            "sanitized_text": sanitized_text,
            "raw_text": raw_text,
            "result": res,                    # original ValidationOutcome (if applicable)
        }
        return result

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
            "success": False,
            "guardrail_name": name,
            "filtered": False,
            "action": "error",
            "reasons": [str(e)],
        }

def run_all_guardrails_sync(
    guardrails: Dict[str, Any],
    input_text: str,
    context_text: str = "",
    llm_output_text: str | None = None,
) -> Dict[str, dict]:
    """
    Run all guardrails sequentially (synchronously).
    Also builds the right payload for LlmRagEvaluator (factuality_guard).
    """
    def build_payload_and_metadata(name: str):
        # Default: pass the plain user text
        payload = input_text
        metadata = None

        if name == "factuality_guard":
            # LlmRagEvaluator expects:
            #   - value (llm_output) as a STRING
            #   - metadata containing original_prompt + context
            llm_answer = llm_output_text or "No answer"
            payload = llm_answer  # <-- llm_output (string)
            metadata = {
                "original_prompt": input_text,
                "context": context_text or "",
            }
        return payload, metadata

    results = {}
    for name, guard in guardrails.items():
        payload, metadata = build_payload_and_metadata(name)
        results[name] = _call_guard_sync(name, guard, payload, metadata)
    return results


async def run_all_guardrails_parallel(
    guardrails: Dict[str, Any],
    input_text: str,
    context_text: str = "",
    llm_output_text: str | None = None,
):
    """
    Run guardrails concurrently, supporting both sync and async .validate().
    Also builds the right payload for LlmRagEvaluator (factuality_guard).
    """

    # --- drop-in replacement for your call_guard inside run_all_guardrails_parallel ---
    async def call_guard(name: str, guard, payload, metadata: dict | None = None):
        try:
            # Most guards accept a single positional 'value'
            if metadata is not None:
                res = guard.validate(payload, metadata=metadata)
            else:
                res = guard.validate(payload)

            # If it's awaitable, await it; otherwise result is already computed
            if inspect.isawaitable(res):
                res = await res

            # Try to interpret Guardrails' ValidationOutcome
            validation_passed = getattr(res, "validation_passed", None)

            # Defaults
            action = "passed"
            reasons = []
            pii_types = []
            sanitized_text = getattr(res, "validated_output", None)
            raw_text = getattr(res, "raw_llm_output", None)

            # Gather summaries (if any)
            summaries = getattr(res, "validation_summaries", []) or []
            failed_any = any(getattr(s, "validator_status", "").lower() == "fail" for s in summaries)

            # Collect human-friendly reasons & PII types from spans
            for s in summaries:
                if getattr(s, "validator_status", "").lower() == "fail":
                    fr = getattr(s, "failure_reason", None)
                    if fr:
                        reasons.append(fr)
                    for span in getattr(s, "error_spans", []) or []:
                        r = getattr(span, "reason", None)
                        if r:
                            reasons.append(r)
                            # crude: many PII guards put the entity name in 'reason'
                            pii_types.append(r)

            # Redaction detection: validated_output differs from raw, or summaries show fails
            redacted = (sanitized_text is not None and raw_text is not None and sanitized_text != raw_text) or failed_any

            # Decide final action:
            # - If the guard signals not passed (e.g., topic filter with on_fail="filter"), mark blocked.
            # - Else if it passed but we see redaction/fails, mark redacted.
            if isinstance(validation_passed, bool):
                if not validation_passed:
                    action = "blocked"
                elif redacted:
                    action = "redacted"
                else:
                    action = "passed"
            else:
                # Non-ValidationOutcome: assume pass
                action = "passed"

            # filtered flag = only when blocked
            filtered = (action == "blocked")

            # Deduplicate reasons & pii_types while preserving order
            def _dedupe(seq):
                seen, out = set(), []
                for x in seq:
                    if x not in seen:
                        out.append(x); seen.add(x)
                return out
            reasons = _dedupe(reasons)
            pii_types = _dedupe(pii_types)

            return {
                "status": "success",
                "success": True,
                "guardrail_name": name,
                "validation_passed": bool(validation_passed) if isinstance(validation_passed, bool) else True,
                "filtered": filtered,
                "action": action,                       # <-- "passed" | "redacted" | "blocked"
                "reasons": reasons,                     # e.g., ["The following text contains PII: ...", "PHONE_NUMBER"]
                "pii_types": pii_types,                 # e.g., ["PHONE_NUMBER"]
                "sanitized_text": sanitized_text,       # useful when action == "redacted"
                "raw_text": raw_text,
                "result": res,                          # keep the outcome object if you need more details
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "success": False,
                "guardrail_name": name,
                "filtered": False,
                "action": "error",
                "reasons": [str(e)],
            }

    def build_payload_and_metadata(name: str):
        # Default: pass the plain user text
        payload = input_text
        metadata = None

        if name == "factuality_guard":
            # LlmRagEvaluator expects:
            #   - value (llm_output) as a STRING
            #   - metadata containing original_prompt + context
            # If you don't have a real model answer yet, provide a test answer.
            llm_answer = llm_output_text or "No answer"
            payload = llm_answer  # <-- llm_output (string)
            metadata = {
                "original_prompt": input_text,
                "context": context_text or "",
            }
        return payload, metadata

    # Create tasks
    tasks = []
    for name, guard in guardrails.items():
        payload, metadata = build_payload_and_metadata(name)
        tasks.append(asyncio.create_task(call_guard(name, guard, payload, metadata)))

    # Gather results
    results_raw = await asyncio.gather(*tasks, return_exceptions=False)
    # Map by guardrail name
    processed = {r["guardrail_name"]: r for r in results_raw}
    return processed