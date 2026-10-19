"""Build a KIE provider by name (used by the app and the benchmarks).

Names:
    rules                  rule baseline, no store list
    rules+gaz              rules + gazetteer of known sellers/addresses (train-derived)
    rules+gaz-nofb         same, without the tallest-line seller fallback
    rules+gaz+adm-nofb     + official place names: address accent repair and address cue
    rules+gaz+admfix-nofb  + place names for accent repair only
    llm:gemini             Gemini over OCR text (model from GEMINI_MODEL, default 2.5 Flash)
    llm:gemini:<model>     explicit model, optionally ":think<N>" for a thinking budget,
                           e.g. llm:gemini:gemini-2.5-flash:think0
    llm:gemini+gaz         Gemini, then seller/address mapped to the gazetteer spelling
    llm:gemini:rerun<T>    same request, answers cached separately under tag T
    lines                  learned span classifier (configs/kie_line_model.*), no store list
    lines+gaz+adm          + gazetteer and official place names (as the rules)
"""

from __future__ import annotations

from ai.kie.base import KIEProvider

KIE_NAMES = (
    "rules",
    "rules+gaz",
    "rules+gaz-nofb",
    "rules+gaz+adm-nofb",
    "llm:gemini",
    "llm:gemini+gaz",
    "llm:gemini+gaz+adm",
    "lines",
    "lines+gaz+adm",
)


def build_kie(name: str) -> KIEProvider:
    from ai.kie.rule_based import RuleBasedKIE

    if name == "rules":
        return RuleBasedKIE()
    if name.startswith("rules+"):
        flags = set(name.removeprefix("rules").removesuffix("-nofb").split("+")[1:])
        if not flags <= {"gaz", "adm", "admfix"}:
            raise ValueError(f"unknown KIE {name!r}; choose from {KIE_NAMES}")
        return RuleBasedKIE(
            gazetteer=_gazetteer() if "gaz" in flags else None,
            seller_fallback=not name.endswith("-nofb"),
            admin=_admin() if flags & {"adm", "admfix"} else None,
            admin_hint="adm" in flags,
        )
    if name == "lines" or name.startswith("lines+"):
        from ai.kie.learned import LineClassifierKIE

        flags = set(name.split("+")[1:])
        if not flags <= {"gaz", "adm"}:
            raise ValueError(f"unknown KIE {name!r}; choose from {KIE_NAMES}")
        return LineClassifierKIE(
            gazetteer=_gazetteer() if "gaz" in flags else None,
            admin=_admin() if "adm" in flags else None,
        )
    if name.startswith("llm:gemini"):
        from ai.kie.gazetteer import Gazetteer
        from ai.kie.llm import LLMKIE, GeminiBackend

        use_adm = name.endswith("+adm")
        name = name.removesuffix("+adm")
        use_gaz = name.endswith("+gaz")
        parts = name.removesuffix("+gaz").split(":")[1:]
        parts[0] = parts[0].removeprefix("gemini")
        model = next((p for p in parts[1:] if not p.startswith(("think", "rerun"))), None)
        think = next((int(p[5:]) for p in parts[1:] if p.startswith("think")), None)
        tag = next((p[5:] for p in parts[1:] if p.startswith("rerun")), "")
        return LLMKIE(
            GeminiBackend(model=model, thinking_budget=think),
            gazetteer=Gazetteer.load() if use_gaz else None,
            admin=_admin() if use_adm else None,
            run_tag=tag,
        )
    raise ValueError(f"unknown KIE {name!r}; choose from {KIE_NAMES}")


def _gazetteer():
    from ai.kie.gazetteer import Gazetteer

    gaz = Gazetteer.load()
    if gaz is None:
        raise FileNotFoundError(
            "configs/kie_gazetteer.json missing: run python -m evaluation.build_gazetteer"
        )
    return gaz


def _admin():
    from ai.normalization.address import AdminUnits

    admin = AdminUnits.load()
    if admin is None:
        raise FileNotFoundError(
            "configs/vn_admin_units.json missing: run python scripts/build_admin_units.py"
        )
    return admin
