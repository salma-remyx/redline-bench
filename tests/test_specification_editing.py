"""Integration tests for the specification-editing metrics.

Ports Vector-Bench's (arxiv:2607.19056v1) "make the requested change
AND leave the rest alone" framing onto RedlineBench's attorney-redline
plumbing: requested edits = paragraphs the attorney touched; protected
body = every other paragraph.

The headline test drives the real wiring — it builds minimal docx
fixtures on disk and calls ``metrics_summary._build_docx_metrics`` (a
NON-NEW module), which now invokes
``specification_editing.compute_specification_editing`` and returns its
result as the third tuple element.
"""

import zipfile
from pathlib import Path

import metrics_summary
import specification_editing

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_TASK = "redline-s1-t1-g01a"  # turn_of() -> 1 (paragraph indices align)


def _build_docx(path: Path, paragraphs: list[list[tuple]]) -> None:
    """Write a minimal docx whose ``word/document.xml`` is built from
    ``paragraphs``. Each paragraph is a list of segments:

      - ``("text", s)``            plain run (paragraph stays protected)
      - ``("ins", author, s)``     insertion revision (paragraph touched)
      - ``("del", author, s)``     deletion revision (paragraph touched)
    """
    body = []
    for segs in paragraphs:
        children = []
        for seg in segs:
            if seg[0] == "text":
                children.append(
                    f'<w:r><w:t xml:space="preserve">{seg[1]}</w:t></w:r>'
                )
            else:
                kind, author, text = seg
                tag = "ins" if kind == "ins" else "del"
                inner = "t" if kind == "ins" else "delText"
                children.append(
                    f'<w:{tag} w:id="1" w:author="{author}" '
                    f'w:date="2026-01-01T00:00:00Z">'
                    f'<w:r><w:{inner} xml:space="preserve">{text}</w:{inner}></w:r>'
                    f'</w:{tag}>'
                )
        body.append("<w:p>" + "".join(children) + "</w:p>")
    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{_W}">' + "".join(body) + "</w:document>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", xml)


# Five paragraphs; the attorney touches indices 1 and 3 → the requested
# edits. Indices 0, 2, 4 are the protected body.
_EXPERT = [
    [("text", "Para 0.")],
    [("text", "Para 1."), ("ins", "Attorney", "edit1")],
    [("text", "Para 2.")],
    [("text", "Para 3."), ("del", "Attorney", "gone")],
    [("text", "Para 4.")],
]


def _seed_run(tmp_path: Path, model: str, paragraphs: list[list[tuple]]) -> None:
    """Lay down one model's turn-1 redline.docx in the layout
    ``find_model_docx_paths`` walks."""
    _build_docx(
        tmp_path / "runs" / "trajectories" / model / _TASK / "redline.docx",
        paragraphs,
    )


def test_build_docx_metrics_wires_specification_signals(tmp_path: Path) -> None:
    """The call site (``_build_docx_metrics``) now returns the
    specification-editing dict and scores it correctly across models."""
    _build_docx(
        tmp_path / "benchmark" / "tasks" / _TASK / "tests" / "attorney_redlines.docx",
        _EXPERT,
    )
    # Good model: touches exactly the attorney's paragraphs 1 and 3.
    _seed_run(
        tmp_path,
        "good-model",
        [
            [("text", "Para 0.")],
            [("text", "Para 1."), ("ins", "Reviewing Counsel", "edit1")],
            [("text", "Para 2.")],
            [("text", "Para 3."), ("ins", "Reviewing Counsel", "edit3")],
            [("text", "Para 4.")],
        ],
    )
    # Sloppy model: also corrupts protected paragraph 2.
    _seed_run(
        tmp_path,
        "sloppy-model",
        [
            [("text", "Para 0.")],
            [("text", "Para 1."), ("ins", "Reviewing Counsel", "edit1")],
            [("text", "Para 2."), ("ins", "Reviewing Counsel", "oops")],
            [("text", "Para 3."), ("ins", "Reviewing Counsel", "edit3")],
            [("text", "Para 4.")],
        ],
    )
    # Lazy model: only addresses paragraph 1 (partial repair).
    _seed_run(
        tmp_path,
        "lazy-model",
        [
            [("text", "Para 0.")],
            [("text", "Para 1."), ("ins", "Reviewing Counsel", "edit1")],
            [("text", "Para 2.")],
            [("text", "Para 3.")],
            [("text", "Para 4.")],
        ],
    )

    _verbosity, _surgicalness, specification = metrics_summary._build_docx_metrics(
        tmp_path / "runs", tmp_path / "benchmark", include_fable_5=False
    )

    # Good model: full specification success, no collateral.
    good = specification["good-model"]
    assert good["repair_progress"] == 1.0
    assert good["unintended_change_rate"] == 0.0
    assert good["specification_success_rate"] == 1.0
    assert good["valid_output_rate"] == 1.0
    assert good["n_tasks"] == 1 and good["n_tasks_valid"] == 1

    # Sloppy model: addressed everything but corrupted 1 of 3 protected
    # paragraphs → high repair progress, nonzero UCR, no spec success.
    sloppy = specification["sloppy-model"]
    assert sloppy["repair_progress"] == 1.0
    assert sloppy["unintended_change_rate"] == round(1 / 3, 4)
    assert sloppy["specification_success_rate"] == 0.0

    # Lazy model: partial repair, no collateral → no spec success.
    lazy = specification["lazy-model"]
    assert lazy["repair_progress"] == 0.5
    assert lazy["unintended_change_rate"] == 0.0
    assert lazy["specification_success_rate"] == 0.0

    # No expert baseline row: the attorney redline IS the spec reference.
    assert "expert" not in specification


def test_invalid_output_gates_reward(tmp_path: Path) -> None:
    """A model redline that isn't a valid docx is validity-gated out of
    the repair/UCR means and earns a 0 specification reward."""
    expert = tmp_path / "expert.docx"
    _build_docx(expert, _EXPERT)
    junk = tmp_path / "model.docx"
    junk.write_text("not a docx")  # BadZipFile on load

    out = specification_editing.compute_specification_editing(
        {"broken-model": [(_TASK, junk, expert)]},
        {_TASK: expert},
    )["broken-model"]
    assert out["valid_output_rate"] == 0.0
    assert out["specification_success_rate"] == 0.0
    assert out["n_tasks"] == 1 and out["n_tasks_valid"] == 0
