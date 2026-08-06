"""Official WebShop split and public-action protocol helpers.

The upstream environment shuffles all 12,087 human goals with seed 233 and
then addresses them by integer session id.  Its official baselines define the
held-out test prefix as ``[0, 500)``, development as ``[500, 1500)``, and
training as ``[1500, 12087)``.  Keeping this mapping in the runtime prevents a
random corpus sample from being mislabeled as an official split.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence


WEBSHOP_TOTAL_HUMAN_GOALS = 12_087
WEBSHOP_OFFICIAL_SPLIT_RANGES = {
    "test": (0, 500),
    "dev": (500, 1_500),
    "train": (1_500, WEBSHOP_TOTAL_HUMAN_GOALS),
}
_SPLIT_ALIASES = {"validation": "dev", "valid": "dev", "eval": "dev"}
_SEARCH_RE = re.compile(r"^search\[(.+)\]$", re.IGNORECASE | re.DOTALL)
_CLICK_RE = re.compile(r"^click\[(.+)\]$", re.IGNORECASE | re.DOTALL)


def canonical_webshop_split(split: str) -> str:
    value = str(split).strip().lower()
    value = _SPLIT_ALIASES.get(value, value)
    if value not in WEBSHOP_OFFICIAL_SPLIT_RANGES:
        raise ValueError(f"unsupported official WebShop split: {split!r}")
    return value


def official_webshop_indices(total_goals: int, split: str) -> tuple[int, ...]:
    """Return the exact upstream session ids for one official split."""

    if int(total_goals) != WEBSHOP_TOTAL_HUMAN_GOALS:
        raise ValueError(
            "official WebShop human-goal corpus must expose exactly "
            f"{WEBSHOP_TOTAL_HUMAN_GOALS} goals after the upstream seed-233 shuffle; "
            f"got {total_goals}"
        )
    start, stop = WEBSHOP_OFFICIAL_SPLIT_RANGES[canonical_webshop_split(split)]
    return tuple(range(start, stop))


def validate_webshop_manifest_indices(
    split: str,
    indices: Iterable[int],
    *,
    complete_official_split: bool,
) -> None:
    canonical = canonical_webshop_split(split)
    observed = tuple(int(index) for index in indices)
    if not observed:
        raise ValueError("WebShop manifest cannot contain an empty split")
    allowed = official_webshop_indices(WEBSHOP_TOTAL_HUMAN_GOALS, canonical)
    allowed_set = set(allowed)
    outside = sorted(set(observed) - allowed_set)
    if outside:
        raise ValueError(
            f"WebShop {canonical} manifest contains session ids outside its official range: "
            f"{outside[:5]}"
        )
    if len(set(observed)) != len(observed):
        raise ValueError("WebShop manifest contains duplicate session ids")
    if complete_official_split and tuple(sorted(observed)) != allowed:
        raise ValueError(
            f"complete WebShop {canonical} manifest must contain exactly {len(allowed)} "
            "official session ids"
        )


@dataclass(frozen=True)
class WebShopActionCheck:
    action: str
    accepted: bool
    feedback: str = ""


def _canonical_valid_actions(valid_actions: Sequence[str]) -> dict[str, str]:
    return {str(action).strip().lower(): str(action).strip() for action in valid_actions}


def check_webshop_action(
    raw_action: str,
    valid_actions: Sequence[str],
    *,
    attempted_actions: Iterable[str] = (),
) -> WebShopActionCheck:
    """Validate one model action using only the visible WebShop action schema."""

    value = str(raw_action).strip()
    if value.lower().startswith("action:"):
        value = value.split(":", 1)[1].strip()
    value = value.splitlines()[0].strip() if value else ""
    valid = _canonical_valid_actions(valid_actions)
    attempted = {str(action).strip().lower() for action in attempted_actions}

    search = _SEARCH_RE.fullmatch(value)
    if search and "search[<keywords>]" in valid:
        keywords = " ".join(search.group(1).split())
        action = f"search[{keywords}]"
    else:
        click = _CLICK_RE.fullmatch(value)
        normalized = f"click[{click.group(1).strip()}]".lower() if click else ""
        action = valid.get(normalized, "")

    if not action:
        examples = ", ".join(tuple(valid.values())[:12])
        return WebShopActionCheck(
            action=value,
            accepted=False,
            feedback=(
                "The proposed action is not currently executable. Return exactly one visible "
                f"search[...] or click[...] action. Current actions include: {examples}"
            ),
        )
    # A WebShop search always re-executes the query and produces a fresh results
    # page, so it can never be "already attempted without progress" even when the
    # agent returns to the (byte-identical) home page and re-searches.  Only the
    # repeat-click heuristic should suppress actions that truly leave state fixed.
    if action.lower() in attempted and not action.lower().startswith("search["):
        return WebShopActionCheck(
            action=action,
            accepted=False,
            feedback=(
                f"{action} was already attempted from this exact page without progress. "
                "Choose a different visible action."
            ),
        )
    return WebShopActionCheck(action=action, accepted=True)


def webshop_fallback_action(
    valid_actions: Sequence[str],
    *,
    goal: str,
    attempted_actions: Iterable[str] = (),
) -> str | None:
    """Choose a deterministic public-state-only fallback after failed retries."""

    valid = _canonical_valid_actions(valid_actions)
    attempted = {str(action).strip().lower() for action in attempted_actions}
    back = valid.get("click[back to search]")
    if back and back.lower() not in attempted:
        return back
    if "search[<keywords>]" in valid:
        keywords = " ".join(str(goal).split())[:400].strip()
        candidate = f"search[{keywords}]" if keywords else ""
        if candidate and candidate.lower() not in attempted:
            return candidate
    low_information = {
        "click[next >]",
        "click[< prev]",
        "click[description]",
        "click[features]",
        "click[reviews]",
    }
    for normalized, action in valid.items():
        if normalized == "search[<keywords>]" or normalized in attempted:
            continue
        if normalized not in low_information:
            return action
    for normalized, action in valid.items():
        if normalized != "search[<keywords>]" and normalized not in attempted:
            return action
    # Last-resort recovery on the home/search page: a search always re-executes
    # the query and yields fresh state, so re-issuing the goal search (even if
    # previously attempted) is legitimate progress rather than a no-progress loop.
    if "search[<keywords>]" in valid:
        keywords = " ".join(str(goal).split())[:400].strip()
        candidate = f"search[{keywords}]" if keywords else ""
        if candidate:
            return candidate
    return None
