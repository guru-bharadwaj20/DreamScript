/**
 * Phase 16.2.7 - the highlighter, and the one property that is not negotiable.
 *
 * This code is **displayed and copied verbatim**. A colour being wrong is a cosmetic mistake; a
 * character being added, dropped or reordered is a person pasting a program that is not the one
 * the pipeline produced. So `tokenise` is total, and the first block below proves it on every line
 * of real emitter output for all four languages: the concatenated token values equal the input,
 * byte for byte.
 *
 * Everything after that is colouring, and it is tested at the level of "does this look like the
 * language" rather than exhaustively - because it is a regex tokeniser and not a parser, and
 * pretending otherwise in a test suite would be claiming a guarantee the code does not make.
 */

import { describe, expect, it } from "vitest";

import { type Language, highlight, languageOf, tokenise } from "./highlight";

/** Real 12.1.6 emitter output, one sample per target. */
const SAMPLES: Record<Language, string> = {
  python: `def run_upload(ctx):
    state = "d000"
    pending = []
    while True:
        if state == "d000":
            ctx = receive_order(ctx)  # a stub the diagram named
            state = "d001"
        else:
            if pending:
                state = pending.pop(0)
                continue
            return ctx
`,
  sql: `-- generated from an ER diagram
CREATE TABLE customer (
    customer_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);
CREATE TABLE "order" (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER REFERENCES customer(customer_id)
);
`,
  react: `export default function ScreenScreen() {
  return (
    <div className="min-h-screen bg-white p-6">
      {"x\\") or 1"}
      {/* a comment */}
    </div>
  );
}
`,
  spice: `* circuit - generated from a circuit diagram
R0 1 0 1k
VSRC_AUTO 1 0 DC 5
.op
.end
`,
};

const joined = (code: string, language: Language) =>
  highlight(code, language)
    .map((line) => line.map((t) => t.value).join(""))
    .join("\n");

describe("tokenise never changes the text", () => {
  it.each(Object.keys(SAMPLES) as Language[])("round-trips real %s output", (language) => {
    expect(joined(SAMPLES[language], language)).toBe(SAMPLES[language]);
  });

  it.each([
    "",
    " ",
    "\t\tindented",
    "an unterminated string: 'oh",
    'a backslash at the end: "\\\\',
    "# just a comment",
    "-- 中文 and emoji 🙂 and an em dash —",
    "a = b == c != d <= e >= f",
    "nested(call(deep(1, 2.5e-3)))",
    "{}[]();:,",
    "'''triple'''",
    "0x1f 1_000 3.14 1e10 5k 10meg",
  ])("round-trips the awkward line %j", (line) => {
    for (const language of Object.keys(SAMPLES) as Language[]) {
      expect(
        tokenise(line, language)
          .map((t) => t.value)
          .join(""),
        `${language}: ${line}`,
      ).toBe(line);
    }
  });

  it("round-trips a line built from every byte that can appear in one", () => {
    // A blunt instrument on purpose: whatever the tokeniser does with these, it must hand them all
    // back. This is the test that would catch an accidental `trim()` or a dropped branch.
    let line = "";
    for (let code = 32; code < 127; code++) line += String.fromCharCode(code);
    for (const language of Object.keys(SAMPLES) as Language[]) {
      expect(tokenise(line, language).map((t) => t.value).join(""), language).toBe(line);
    }
  });

  it("emits no empty tokens, which would be noise in the DOM", () => {
    for (const language of Object.keys(SAMPLES) as Language[]) {
      for (const line of highlight(SAMPLES[language], language)) {
        for (const token of line) expect(token.value.length).toBeGreaterThan(0);
      }
    }
  });
});

describe("colouring", () => {
  const kinds = (line: string, language: Language) =>
    Object.fromEntries(
      tokenise(line, language).map((t) => [t.value, t.kind] as const),
    ) as Record<string, string>;

  it("finds python keywords, strings, numbers and calls", () => {
    const k = kinds('    if state == "d000": go(1.5)  # note', "python");
    expect(k["if"]).toBe("keyword");
    expect(k['"d000"']).toBe("string");
    expect(k["1.5"]).toBe("number");
    expect(k["go"]).toBe("function");
    expect(k["# note"]).toBe("comment");
  });

  it("is case-insensitive for SQL, because emitted DDL is upper and hand-written is not", () => {
    expect(kinds("create table t (id integer);", "sql")["create"]).toBe("keyword");
    expect(kinds("CREATE TABLE t (id INTEGER);", "sql")["CREATE"]).toBe("keyword");
  });

  it("treats a leading star as a SPICE comment only in column zero", () => {
    // `*` is multiplication everywhere else on the line, and a netlist uses it.
    expect(tokenise("* a title line", "spice")[0].kind).toBe("comment");
    expect(kinds("R1 1 0 2*k", "spice")["*"]).toBe("punct");
  });

  it("stops a string at the end of its line rather than letting it run away", () => {
    // The failure mode of a naive highlighter: one unterminated quote colours the rest of the
    // file, and a person stops trusting what they are copying.
    const first = tokenise("x = 'unterminated", "python");
    expect(first[first.length - 1].kind).toBe("string");
    const second = tokenise("y = 2", "python");
    expect(second.some((t) => t.kind === "string")).toBe(false);
  });

  it("honours a backslash escape inside a string", () => {
    const line = 'x = "a\\"b" + y';
    const string = tokenise(line, "python").find((t) => t.kind === "string");
    expect(string?.value).toBe('"a\\"b"');
  });
});

describe("languageOf", () => {
  it("maps the four the emitters produce and refuses anything else", () => {
    expect(languageOf("python")).toBe("python");
    expect(languageOf("SQL")).toBe("sql");
    expect(languageOf("react")).toBe("react");
    expect(languageOf("spice")).toBe("spice");
    expect(languageOf("rust")).toBeNull();
    expect(languageOf(null)).toBeNull();
    expect(languageOf(undefined)).toBeNull();
  });
});
