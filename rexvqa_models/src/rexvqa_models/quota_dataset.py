import hashlib
import logging
from collections import Counter
from typing import Any, cast

import pandas as pd

from rexvqa_models.image_utils import normalize_list_field
from rexvqa_models.io_utils import load_json, write_json
from rexvqa_models.types import CaseData, PathLike

CategoryQuotas = dict[str, int]
CategoryDistribution = dict[str, float]
CaseMap = dict[str, CaseData]
OTHER_CATEGORY = "other"

LOGGER = logging.getLogger(__name__)


def normalize_validation_code(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def stable_case_sort_key(case_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).hexdigest()


def normalize_category_distribution(
    category_distribution: dict[str, Any],
) -> CategoryDistribution:
    normalized = {
        str(category): float(value)
        for category, value in category_distribution.items()
        if float(value) > 0
    }
    if not normalized:
        raise ValueError(
            "At least one positive category distribution weight is required."
        )

    total_weight = sum(normalized.values())
    return {category: weight / total_weight for category, weight in normalized.items()}


def _allocate_integer_counts(
    total: int,
    weights: CategoryDistribution,
    capacities: CategoryQuotas,
) -> CategoryQuotas:
    if total <= 0:
        raise ValueError("target_total must be positive.")
    if total > sum(capacities.values()):
        raise ValueError(
            f"Requested target_total {total} exceeds eligible capacity {sum(capacities.values())}."
        )

    allocations: dict[str, int] = {category: 0 for category in weights}
    remaining = total

    while remaining > 0:
        active = {
            cat: weights[cat] for cat in weights if allocations[cat] < capacities[cat]
        }
        if not active:
            raise ValueError(f"Unable to allocate remaining {remaining} cases.")

        scale = remaining / sum(active.values())
        ideal = {cat: w * scale for cat, w in active.items()}

        for cat, v in ideal.items():
            add = min(int(v), capacities[cat] - allocations[cat])
            allocations[cat] += add
            remaining -= add

        if remaining <= 0:
            break

        # Distribute remaining units by largest fractional part first
        for cat in sorted(active, key=lambda c: (-(ideal[c] % 1), c)):
            if remaining <= 0:
                break
            if allocations[cat] < capacities[cat]:
                allocations[cat] += 1
                remaining -= 1

    return {cat: count for cat, count in allocations.items() if count > 0}


def _build_category_quotas(
    frame: pd.DataFrame,
    target_total: int,
    category_distribution: dict[str, Any],
) -> CategoryQuotas:
    normalized_distribution = normalize_category_distribution(category_distribution)
    available_counts = {
        str(cat): int(count)
        for cat, count in frame["category"].value_counts().to_dict().items()
    }

    explicit_distribution = {
        cat: w for cat, w in normalized_distribution.items() if cat != OTHER_CATEGORY
    }
    missing = sorted(
        cat for cat in explicit_distribution if cat not in available_counts
    )
    if missing:
        raise ValueError(
            f"Requested categories are not available after filtering: {', '.join(missing)}."
        )

    other_weight = normalized_distribution.get(OTHER_CATEGORY, 0.0)
    remaining_categories = [
        cat for cat in available_counts if cat not in explicit_distribution
    ]
    expanded_distribution = dict(explicit_distribution)
    if other_weight > 0 and remaining_categories:
        shared_weight = other_weight / len(remaining_categories)
        expanded_distribution.update(
            {cat: shared_weight for cat in remaining_categories}
        )

    if not expanded_distribution:
        raise ValueError(
            "No eligible categories remain after applying category_distribution."
        )

    normalized = normalize_category_distribution(expanded_distribution)
    capacities = {cat: available_counts[cat] for cat in normalized}
    return _allocate_integer_counts(
        total=target_total, weights=normalized, capacities=capacities
    )


def _cases_to_frame(cases: CaseMap, seed: int) -> pd.DataFrame:
    frame = pd.DataFrame.from_dict(cases, orient="index")
    frame.index.name = "case_id"
    frame = frame.reset_index()
    frame["case_id"] = frame["case_id"].astype(str)
    frame["category"] = frame["category"].fillna("").astype(str)
    frame["subcategory"] = frame["subcategory"].fillna("").astype(str)
    frame["study_id"] = frame["study_id"].fillna("").astype(str)
    frame["validation_code"] = frame["validation_code"].map(normalize_validation_code)
    frame["image_count"] = frame["ImagePath"].map(normalize_list_field).map(len)
    frame["stable_key"] = frame["case_id"].map(
        lambda case_id: stable_case_sort_key(case_id, seed)
    )
    return frame


def summarize_cases(cases: CaseMap) -> dict[str, Any]:
    frame = _cases_to_frame(cases, seed=0)
    subcategory_counts = (
        frame.value_counts(["category", "subcategory"])
        .head(25)
        .reset_index(name="count")
    )

    return {
        "num_cases": len(frame),
        "category_counts": frame["category"].value_counts().to_dict(),
        "top_subcategories": [
            {
                "category": str(row["category"]),
                "subcategory": str(row["subcategory"]),
                "count": int(row["count"]),
            }
            for row in subcategory_counts.to_dict("records")
        ],
        "study_question_histogram": frame["study_id"]
        .value_counts()
        .value_counts()
        .to_dict(),
        "image_count_histogram": frame["image_count"].value_counts().to_dict(),
    }


def _select_from_ordered_frame(
    frame: pd.DataFrame,
    quotas: CategoryQuotas,
    max_cases_per_study: int,
) -> pd.DataFrame:
    selected_case_id_set: set[str] = set()
    selected_per_category: Counter[str] = Counter()
    study_counts: Counter[str] = Counter()
    category_subcategory_counts: Counter[tuple[str, str]] = Counter()

    # First pass: respect subcategory cap
    for row in frame[
        ["case_id", "category", "subcategory", "study_id", "max_cases_per_subcategory"]
    ].itertuples(index=False, name=None):
        case_id, category, subcategory, study_id, max_cases_per_subcategory = row
        case_id = str(case_id)
        category = str(category)
        subcategory = str(subcategory)
        study_id = str(study_id)

        if case_id in selected_case_id_set:
            continue
        if selected_per_category[category] >= quotas[category]:
            continue
        if study_counts[study_id] >= max_cases_per_study:
            continue
        if (
            category_subcategory_counts[(category, subcategory)]
            >= max_cases_per_subcategory
        ):
            continue

        selected_case_id_set.add(case_id)
        selected_per_category[category] += 1
        study_counts[study_id] += 1
        category_subcategory_counts[(category, subcategory)] += 1

    # Second pass: fill remaining quota without subcategory cap
    for row in frame[
        ["case_id", "category", "subcategory", "study_id", "max_cases_per_subcategory"]
    ].itertuples(index=False, name=None):
        case_id, category, subcategory, study_id, _ = row
        case_id = str(case_id)
        category = str(category)
        study_id = str(study_id)

        if case_id in selected_case_id_set:
            continue
        if selected_per_category[category] >= quotas[category]:
            continue
        if study_counts[study_id] >= max_cases_per_study:
            continue

        selected_case_id_set.add(case_id)
        selected_per_category[category] += 1
        study_counts[study_id] += 1

    return frame[frame["case_id"].isin(selected_case_id_set)]


def build_quota_subset(
    cases: CaseMap,
    target_total: int,
    category_distribution: dict[str, Any],
    max_cases_per_study: int = 2,
    max_subcategory_fraction: float = 0.4,
    seed: int = 0,
) -> CaseMap:
    frame = _cases_to_frame(cases, seed=seed)
    frame = frame[
        frame["validation_code"].isin({"0", "1"}) & frame["image_count"].isin({1, 2})
    ].copy()

    category_quotas = _build_category_quotas(
        frame=frame,
        target_total=int(target_total),
        category_distribution=category_distribution,
    )
    frame = frame[frame["category"].isin(category_quotas)].copy()

    frame["quota"] = frame["category"].map(category_quotas).astype(int)
    frame["max_cases_per_subcategory"] = frame["quota"].map(
        lambda quota: max(1, int(quota * max_subcategory_fraction))
    )

    frame = frame.sort_values(
        ["category", "subcategory", "stable_key"],
        kind="mergesort",
    )

    selected = _select_from_ordered_frame(
        frame=frame,
        quotas=category_quotas,
        max_cases_per_study=max_cases_per_study,
    )

    selected_counts = selected["category"].value_counts().to_dict()
    shortfall = sum(
        category_quotas[cat] - int(selected_counts.get(cat, 0))
        for cat in category_quotas
    )
    if shortfall > 0:
        for category, quota in category_quotas.items():
            selected_count = int(selected_counts.get(category, 0))
            if selected_count < quota:
                LOGGER.warning(
                    "Category quota shortfall for '%s': selected %d of %d.",
                    category,
                    selected_count,
                    quota,
                )

        # Backfill shortfall from unselected cases across all categories, respecting study cap
        remaining = frame[~frame["case_id"].isin(selected["case_id"])].copy()
        backfill_quotas = {cat: shortfall for cat in remaining["category"].unique()}
        backfill = _select_from_ordered_frame(
            frame=remaining,
            quotas=backfill_quotas,
            max_cases_per_study=max_cases_per_study,
        )
        backfill = backfill.head(shortfall)
        selected = pd.concat([selected, backfill], ignore_index=True)
        LOGGER.warning(
            "Backfilled %d case(s) from other categories to reach target total.",
            len(backfill),
        )

    actual_total = len(selected)
    if actual_total != target_total:
        LOGGER.warning(
            "Final subset has %d cases (target was %d); dataset is too constrained.",
            actual_total,
            target_total,
        )

    selected_ids = sorted(cast(list[str], selected["case_id"].tolist()))
    return {case_id: cases[case_id] for case_id in selected_ids}


def build_quota_subset_file(
    input_file: PathLike,
    output_file: PathLike,
    target_total: int,
    category_distribution: dict[str, Any],
    summary_file: PathLike | None = None,
    **selection_kwargs: Any,
) -> CaseMap:
    cases = load_json(input_file)
    if not isinstance(cases, dict):
        raise ValueError(f"Expected a JSON object of cases in {input_file}.")

    subset = build_quota_subset(
        cases=cases,
        target_total=target_total,
        category_distribution=category_distribution,
        **selection_kwargs,
    )
    write_json(subset, output_file)

    if summary_file:
        write_json(summarize_cases(subset), summary_file)

    return subset
