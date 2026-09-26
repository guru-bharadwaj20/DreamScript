/**
 * Phase 16.2.7 - syntax highlighting for four languages, in about a hundred lines.
 *
 * ## Why not Prism, Shiki or highlight.js
 *
 * Shiki is a full TextMate grammar engine and ships megabytes. highlight.js is ~45 kB for its core
 * plus a language each. Prism is the light one at ~2 kB core, and its Python, SQL and JSX grammars
 * come to about 12 kB more - against a whole bundle of 190 kB that 16.3.1 caches for offline use.
 *
 * What is being highlighted here is not arbitrary source. It is **12.1.6's emitter output**: a few
 * hundred bytes of generated Python, SQL, JSX or SPICE, in a narrow and known style. A tokeniser
 * that handles strings, comments, numbers and a keyword list is the whole requirement.
 *
 * **And the honest limits.** This is a regex tokeniser, not a parser. It does not track nesting, it
 * does not know a template literal from a string, and it will mis-colour a keyword used as an
 * identifier (`class = 1`). Every one of those is a *colour* being wrong on code that is displayed
 * and copied verbatim, which is a different order of mistake from a parser being wrong. What it must
 * never do is change the text, and `tokenise` is total: the concatenated token values equal the
 * input, byte for byte, which is the one property that is tested exhaustively.
 */

export type TokenKind = "plain" | "keyword" | "string" | "comment" | "number" | "function" | "punct";

export interface Token {
  kind: TokenKind;
  value: string;
}

export type Language = "python" | "sql" | "react" | "spice";

const KEYWORDS: Record<Language, string[]> = {
  python: [
    "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del", "elif",
    "else", "except", "False", "finally", "for", "from", "global", "if", "import", "in", "is",
    "lambda", "None", "nonlocal", "not", "or", "pass", "raise", "return", "True", "try", "while",
    "with", "yield", "self",
  ],
  sql: [
    "ALTER", "AND", "AS", "BY", "CASCADE", "CHECK", "CONSTRAINT", "CREATE", "DEFAULT", "DELETE",
    "DROP", "FOREIGN", "FROM", "GROUP", "INDEX", "INSERT", "INTO", "JOIN", "KEY", "NOT", "NULL",
    "ON", "OR", "ORDER", "PRIMARY", "REFERENCES", "SELECT", "SET", "TABLE", "UNIQUE", "UPDATE",
    "VALUES", "VIEW", "WHERE", "INTEGER", "TEXT", "REAL", "BLOB", "VARCHAR", "BOOLEAN", "DATE",
  ],
  react: [
    "as", "async", "await", "break", "case", "catch", "class", "const", "continue", "default",
    "delete", "do", "else", "export", "extends", "false", "finally", "for", "from", "function",
    "if", "import", "in", "instanceof", "let", "new", "null", "of", "return", "super", "switch",
    "this", "throw", "true", "try", "typeof", "undefined", "var", "void", "while", "yield",
  ],
  spice: [
    ".ac", ".dc", ".end", ".ends", ".ic", ".include", ".lib", ".model", ".op", ".options",
    ".param", ".plot", ".print", ".subckt", ".temp", ".tran", "AC", "DC", "PULSE", "SIN",
  ],
};

/** `language_for` in `src/codegen/targets.py` maps a diagram type to one of these. */
export function languageOf(name: string | null | undefined): Language | null {
  const value = String(name ?? "").toLowerCase();
  if (value === "python" || value === "sql" || value === "react" || value === "spice") return value;
  return null;
}

/**
 * One line of source into coloured spans.
 *
 * Per line rather than over the whole program, so a block comment cannot run away and colour the
 * rest of the file - which is the failure mode of a naive tokeniser and the reason a person copies
 * code out of a highlighter that has given up. None of the four languages here emits a multi-line
 * string in generated output, so nothing correct is lost.
 */
export function tokenise(line: string, language: Language): Token[] {
  const out: Token[] = [];
  const keywords = new Set(KEYWORDS[language].map((k) => (language === "sql" ? k : k)));
  const caseless = language === "sql" || language === "spice";
  let i = 0;
  let plain = "";

  const flush = () => {
    if (plain) out.push({ kind: "plain", value: plain });
    plain = "";
  };
  const push = (kind: TokenKind, value: string) => {
    flush();
    out.push({ kind, value });
  };

  while (i < line.length) {
    const rest = line.slice(i);

    // Comments, to end of line. SPICE is the odd one: a leading `*` is a comment only in column 0.
    const comment =
      (language === "python" && rest.startsWith("#")) ||
      (language === "sql" && rest.startsWith("--")) ||
      (language === "react" && rest.startsWith("//")) ||
      (language === "spice" && (i === 0 ? rest.startsWith("*") : rest.startsWith(";")));
    if (comment) {
      push("comment", rest);
      i = line.length;
      continue;
    }

    const quote = rest[0];
    if (quote === '"' || quote === "'" || quote === "`") {
      // Scan to the closing quote, honouring a backslash escape. An unterminated string takes the
      // rest of the line and stops there - it cannot leak into the next one.
      let j = 1;
      while (j < rest.length) {
        if (rest[j] === "\\") j += 2;
        else if (rest[j] === quote) {
          j += 1;
          break;
        } else j += 1;
      }
      push("string", rest.slice(0, Math.min(j, rest.length)));
      i += Math.min(j, rest.length);
      continue;
    }

    const number = /^\d+(\.\d+)?([eE][-+]?\d+)?[a-zA-Z]*/.exec(rest);
    if (number && !/[A-Za-z_]/.test(line[i - 1] ?? "")) {
      push("number", number[0]);
      i += number[0].length;
      continue;
    }

    const word = /^[A-Za-z_.][\w.]*/.exec(rest);
    if (word) {
      const value = word[0];
      const key = caseless ? value.toUpperCase() : value;
      const isKeyword = caseless
        ? [...keywords].some((k) => k.toUpperCase() === key)
        : keywords.has(value);
      if (isKeyword) push("keyword", value);
      // A name followed by `(` is a call. Cheap, and it is most of what makes generated code
      // readable at a glance: the stubs a flowchart calls are the interesting part of it.
      else if (rest.slice(value.length).trimStart().startsWith("(")) push("function", value);
      else plain += value;
      i += value.length;
      continue;
    }

    if (/[{}()[\];:,=<>+\-*/%&|!]/.test(rest[0])) {
      push("punct", rest[0]);
      i += 1;
      continue;
    }

    plain += rest[0];
    i += 1;
  }
  flush();
  return out;
}

/** Every line of a program, tokenised. */
export function highlight(code: string, language: Language): Token[][] {
  return code.split("\n").map((line) => tokenise(line, language));
}
