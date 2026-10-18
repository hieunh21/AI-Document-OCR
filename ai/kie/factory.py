"""Build a KIE provider by name (used by the app and the benchmarks).

Names:
    rules                  rule baseline, no store list
    rules+gaz              rules + gazetteer of known sellers/addresses (train-derived)
    rules+gaz-nofb         same, without the tallest-line seller fallback
    llm:gemini             Gemini over OCR text (model from GEMINI_MODEL, default 2.5 Flash)
    llm:gemini:<model>     explicit model, optionally ":think<N>" for a thinking budget,
                           e.g. llm:gemini:gemini-2.5-flash:think0
    llm:gemini+gaz         Gemini, then seller/address mapped to the gazetteer spelling
"""

from __future__ import annotations

from ai.kie.base import KIEProvider

KIE_NAMES = ("rules", "rules+gaz", "rules+gaz-nofb", "llm:gemini", "llm:gemini+gaz")


def build_kie(name: str) -> KIEProvider:
    from ai.kie.rule_based import RuleBasedKIE

    if name == "rules":
        return RuleBasedKIE()
    if name in ("rules+gaz", "rules+gaz-nofb"):
        from ai.kie.gazetteer import Gazetteer

        gaz = Gazetteer.load()
        if gaz is None:
            raise FileNotFoundError(
                "configs/kie_gazetteer.json missing: run python -m evaluation.build_gazetteer"
            )
        return RuleBasedKIE(gazetteer=gaz, seller_fallback=not name.endswith("-nofb"))
    if name.startswith("llm:gemini"):
        from ai.kie.gazetteer import Gazetteer
        from ai.kie.llm import LLMKIE, GeminiBackend

        use_gaz = name.endswith("+gaz")
        parts = name.removesuffix("+gaz").split(":")[1:]
        parts[0] = parts[0].removeprefix("gemini")
        model = next((p for p in parts[1:] if not p.startswith("think")), None)
        think = next((int(p[5:]) for p in parts[1:] if p.startswith("think")), None)
        return LLMKIE(
            GeminiBackend(model=model, thinking_budget=think),
            gazetteer=Gazetteer.load() if use_gaz else None,
        )
    raise ValueError(f"unknown KIE {name!r}; choose from {KIE_NAMES}")
