"""Phase 12.1.5 - HTML -> React conversion, checked by building and rendering in node."""

from __future__ import annotations

import pytest

from src.codegen import schema
from src.codegen import sketch2code as S
from src.eval import react

NODE = pytest.mark.skipif(not react.toolchain_ready(), reason="needs node + esbuild + react")

PAGE = """<!DOCTYPE html><html><head><title>T</title><style>p{color:red}</style>
<script>var x = "<div>";</script></head>
<body onload="init()">
  <div class="hero" style="background-color: #fff; margin-top:4px" data-id="7" :class="{a: b}">
    <h1>Hello <b>big</b> world</h1>
    <p>First para
    <div>block inside p</div>
    <label for="q">Search &amp; find {braces} "quotes"</label>
    <input id="q" type="text" disabled onclick="go()">
    <img src="rick.jpg" alt="x">
    <br>
  </div>
  <span>unclosed
</body></html>"""


def test_attributes_are_translated_and_unsafe_ones_dropped():
    code, stats = S.html_to_jsx(PAGE)
    assert 'className={"hero"}' in code and 'htmlFor={"q"}' in code
    assert 'style={{"backgroundColor": "#fff", "marginTop": "4px"}}' in code
    assert 'data-id={"7"}' in code and "disabled={true}" in code
    assert "onclick" not in code.lower() and ":class" not in code
    assert stats["dropped_attr:event"] == 1 and stats["dropped_attr:invalid_name"] == 1


def test_void_elements_self_close_and_script_style_head_are_gone():
    code, _ = S.html_to_jsx(PAGE)
    assert "<br />" in code and "<img " in code and "/>" in code
    assert "var x" not in code and "color:red" not in code and "<title>" not in code


def test_text_is_a_string_literal_with_boundary_spaces_kept():
    code, _ = S.html_to_jsx(PAGE)
    assert '{"Search & find {braces} \\"quotes\\""}' in code
    assert '{"Hello "}' in code and '{" world"}' in code


def test_p_is_closed_before_a_block_and_unclosed_elements_are_closed():
    code_closed, stats = S.html_to_jsx(PAGE)
    assert stats["implicit_p_close"] == 1
    assert code_closed.rstrip().endswith("}") and "</span>" in code_closed


@NODE
def test_converted_page_builds_renders_and_keeps_its_text():
    code, _ = S.html_to_jsx(PAGE)
    result = react.check_many([code], keep_html=True)[0]
    assert result["ok"], result
    fidelity = S.text_fidelity(PAGE, result["html"])
    assert fidelity["lost"] == 0 and fidelity["invented"] == 0
    assert fidelity["text_content_equal"]


def test_text_content_sees_joined_words_that_word_counts_miss():
    source = "<body><p>Hello <b>big</b> world</p></body>"
    joined = "<div><p>Hello<b>big</b>world</p></div>"
    fidelity = S.text_fidelity(source, joined)
    assert fidelity["lost"] == 0 and fidelity["invented"] == 0  # blind
    assert not fidelity["text_content_equal"]  # not blind


def test_dropped_text_is_detected():
    fidelity = S.text_fidelity("<body><p>a b c</p></body>", "<div><p>a b</p></div>")
    assert fidelity["lost"] == 1 and fidelity["recall"] < 1


@pytest.mark.skipif(not (S.OUT / "pairs.jsonl").is_file(), reason="needs sketch2code build")
def test_written_shard_is_schema_valid_and_outside_the_training_merge():
    from src.codegen import pairs

    records = S.load_pairs()
    assert len(records) == 484
    assert all(schema.validate(r) == [] for r in records)
    assert not (pairs.PAIRS_DIR / "sketch2code_html.jsonl").exists()
