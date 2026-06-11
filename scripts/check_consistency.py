#!/usr/bin/env python3
"""Cross-file consistency guard for the fine-tuning pipeline.

The capstone Failure Log (FL-FT-01 / FL-FT-02) requires these invariants to be
enforced by an automated check, not by comments:

  1. The custom chat template string is byte-identical between
     finetune_descriptions.ipynb and evaluate_descriptions.ipynb
     (a one-character difference skews every eval prompt — FL-FT-01).
  2. split_uids() carries identical logic in build_metadata_store.ipynb and
     finetune_descriptions.ipynb, golden pinning and the non-golden test floor
     included (the §8 preview must equal the real split — FL-FT-02).
  3. The embedded prompt strings in finetune_descriptions.ipynb match this
     repo's prompts/*.md copies byte-for-byte — and the Improvement Tool
     checkout's prompts/*.md too, when a checkout is present.

Stdlib only; no GPU, no installs. Run from anywhere:

    python scripts/check_consistency.py
"""

import ast
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

failures = []


def check(label, ok):
    print(("PASS" if ok else "FAIL"), "-", label)
    if not ok:
        failures.append(label)


def code_cells(nb_name):
    nb = json.loads((ROOT / nb_name).read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def extract_one(pattern, nb_name, what):
    hits = [
        m.group(1)
        for src in code_cells(nb_name)
        for m in re.finditer(pattern, src, re.S)
    ]
    if len(hits) != 1:
        sys.exit(f"FAIL - expected exactly one {what} in {nb_name}, found {len(hits)}")
    return hits[0]


# ── 1. chat template parity (FL-FT-01) ──────────────────────────────────────
TEMPLATE_RE = r'tokenizer\.chat_template = "((?:[^"\\]|\\.)*)"'
t_train = extract_one(TEMPLATE_RE, "finetune_descriptions.ipynb", "chat template")
t_eval = extract_one(TEMPLATE_RE, "evaluate_descriptions.ipynb", "chat template")
check("chat template byte-identical across notebooks", t_train == t_eval)
check("chat template emits eos_token after assistant turns", "eos_token" in t_train)
check("chat template owns BOS", t_train.startswith("{{ bos_token }}"))

# ── 2. split_uids parity (FL-FT-02) ─────────────────────────────────────────
SPLIT_RE = r"(def split_uids\(.*?\n    return \{[^\n]*\}\n?)"
s_build = extract_one(SPLIT_RE, "build_metadata_store.ipynb", "split_uids def")
s_train = extract_one(SPLIT_RE, "finetune_descriptions.ipynb", "split_uids def")


def normalized_fn(src):
    """AST dump of the function with its docstring removed (comments are
    already invisible to the AST), so doc-wording differences don't matter."""
    fn = ast.parse(src).body[0]
    if (
        fn.body
        and isinstance(fn.body[0], ast.Expr)
        and isinstance(fn.body[0].value, ast.Constant)
        and isinstance(fn.body[0].value.value, str)
    ):
        fn.body = fn.body[1:]
    return ast.dump(fn)


check(
    "split_uids logic identical across notebooks",
    normalized_fn(s_build) == normalized_fn(s_train),
)


# ── 3. embedded prompts == prompt files ──────────────────────────────────────
def extract_embedded(name):
    src = extract_one(rf"{name} = \((.*?)\n\)\n", "finetune_descriptions.ipynb", name)
    return ast.literal_eval("(" + src + "\n)")


embedded = {
    "system.md": extract_embedded("_EMBEDDED_SYSTEM_MD"),
    "column.md": extract_embedded("_EMBEDDED_COLUMN_MD"),
}

for fname, text in embedded.items():
    local = ROOT / "prompts" / fname
    check(
        f"embedded prompt == prompts/{fname}",
        local.exists() and text == local.read_text(encoding="utf-8"),
    )

imp = Path(
    os.environ.get("IMPROVEMENT_TOOL_DIR", ROOT.parent / "AI-Metadata-Improvement-Tool")
)
if (imp / "prompts").is_dir():
    for fname, text in embedded.items():
        check(
            f"embedded prompt == Improvement Tool prompts/{fname}",
            text == (imp / "prompts" / fname).read_text(encoding="utf-8"),
        )
else:
    print(f"SKIP - Improvement Tool checkout not found at {imp}")

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all consistency checks passed")
