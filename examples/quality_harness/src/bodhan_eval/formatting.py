"""Conservative structural checks. Offsets always refer to original strings.

Supports pipe tables and common LaTeX math delimiters, braces, and environments.
This is not a Markdown renderer or TeX compiler. It does not judge equations.
"""

import re


def outside_code(text):
    """Mask fenced and inline code without changing offsets or line breaks."""
    chars = list(text)
    fence = None
    offset = 0
    for line in text.splitlines(keepends=True):
        marker = re.match(r" {0,3}(`{3,}|~{3,})(.*)", line)
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = None
            chars[offset : offset + len(line)] = ["\n" if c == "\n" else " " for c in line]
        elif marker:
            fence = marker[1]
            chars[offset : offset + len(line)] = ["\n" if c == "\n" else " " for c in line]
        offset += len(line)
    masked = "".join(chars)
    for match in re.finditer(r"(`+)(?!`)(.*?)(?<!`)\1(?!`)", masked, re.DOTALL):
        chars[match.start() : match.end()] = ["\n" if c == "\n" else "x" for c in match[0]]
    return "".join(chars)


def anchor(text, field, start, end):
    return {"field": field, "span": [start, end], "text": text[start:end]}


def finding(text, field, start, end, name, reason):
    return {
        "category": "format",
        "severity": "minor",
        "check": name,
        "field": field,
        "span": [start, end],
        "evidence": {"reason": reason, "anchor": anchor(text, field, start, end)},
    }


def table_rows(text):
    rows = []
    offset = 0
    for line in outside_code(text).splitlines(keepends=True):
        # An escaped pipe is cell content. Paired backslashes leave a real pipe.
        visible = re.sub(r"\\.", "xx", line.rstrip("\r\n"))
        stripped = visible.strip()
        cells = stripped.removeprefix("|").removesuffix("|").split("|")
        rows.append((offset, offset + len(line.rstrip("\r\n")), cells, "|" in visible))
        offset += len(line)
    return rows


def tables(text):
    rows = table_rows(text)
    found, errors = [], []
    i = 0
    while i + 1 < len(rows):
        header, separator = rows[i], rows[i + 1]
        cells = separator[2]
        if not header[3] or not separator[3] or not any(re.fullmatch(r"\s*:?-{3,}:?\s*", c) for c in cells):
            i += 1
            continue
        width = len(header[2])
        if len(cells) != width or not all(re.fullmatch(r"\s*:?-{3,}:?\s*", c) for c in cells):
            errors.append((separator[0], separator[1], "table separator does not match header columns"))
        j = i + 2
        while j < len(rows) and rows[j][3]:
            if len(rows[j][2]) != width:
                errors.append((rows[j][0], rows[j][1], "table row has a different column count"))
            j += 1
        found.append((header[0], rows[j - 1][1], width))
        i = j
    return found, errors


def check_tables(source, translation):
    if translation is None:
        return {"status": "unavailable", "reason": "no usable translation"}
    before, source_errors = tables(source)
    after, errors = tables(translation)
    if source_errors:
        return {"status": "inconclusive", "reason": "source table structure is malformed"}
    candidates = [finding(translation, "translation", a, b, "markdown_table", reason) for a, b, reason in errors]
    if before and not after:
        a, b, _ = before[0]
        candidates.append(finding(source, "source", a, b, "markdown_table", "source table structure absent from translation"))
    if before and after and [x[2] for x in before] != [x[2] for x in after]:
        a, b, _ = after[0]
        candidates.append(finding(translation, "translation", a, b, "markdown_table", "table count or column structure changed"))
    if not before and not after:
        return {"status": "skipped", "reason": "no supported pipe tables"}
    return {"status": "evaluated", "verdict": "finding" if candidates else "match", "candidates": candidates}


LATEX = re.compile(r"\\(?:begin|end)\{[^{}]+\}|\\[\[\]()]|\\.|\$\$|\$|[{}]")


def latex_structure(text):
    masked = outside_code(text)
    stack, errors, signature = [], [], []
    for match in LATEX.finditer(masked):
        token = match[0]
        start, end = match.span()
        if token.startswith((r"\begin{", r"\end{")):
            name = token[token.index("{") + 1 : -1]
            # Ignore non-math document environments.
            if name not in {"equation", "equation*", "align", "align*", "aligned", "gather", "gather*", "matrix", "pmatrix", "bmatrix", "cases", "split"}:
                continue
            if token.startswith(r"\begin"):
                stack.append(("env:" + name, start, end))
                signature.append("env:" + name)
            elif stack and stack[-1][0] == "env:" + name:
                stack.pop()
            else:
                errors.append((start, end, "unmatched LaTeX environment end"))
        elif token in {"$", "$$", r"\(", r"\["}:
            # A lone dollar followed by a digit may be a currency amount.
            if token == "$" and not stack and masked[end : end + 1].isdigit():
                if "$" not in masked[end:].split("\n", 1)[0]:
                    continue
            if stack and stack[-1][0] == token and token in {"$", "$$"}:
                stack.pop()
            else:
                stack.append((token, start, end))
                signature.append("math")
        elif token in {r"\)", r"\]"}:
            expected = r"\(" if token == r"\)" else r"\["
            if stack and stack[-1][0] == expected:
                stack.pop()
            else:
                errors.append((start, end, "unmatched LaTeX math delimiter"))
        elif token == "{" and stack:
            stack.append(("{", start, end))
        elif token == "}" and stack:
            if stack[-1][0] == "{":
                stack.pop()
            else:
                errors.append((start, end, "unmatched brace inside LaTeX math"))
    errors.extend((a, b, "unclosed LaTeX delimiter, brace, or environment") for _, a, b in stack)
    return signature, errors


def check_latex(source, translation):
    if translation is None:
        return {"status": "unavailable", "reason": "no usable translation"}
    expected, source_errors = latex_structure(source)
    observed, errors = latex_structure(translation)
    if source_errors:
        return {"status": "inconclusive", "reason": "source LaTeX structure is malformed"}
    candidates = [finding(translation, "translation", a, b, "latex", reason) for a, b, reason in errors]
    if expected and not observed and not errors:
        return {"status": "inconclusive", "reason": "source math markup absent; correspondence needs semantic review"}
    if not expected and not observed and not errors:
        return {"status": "skipped", "reason": "no supported LaTeX math"}
    return {"status": "evaluated", "verdict": "finding" if candidates else "match", "candidates": candidates}


def valid_format_finding(candidate, item):
    field = candidate["field"]
    evidence = candidate["evidence"]["anchor"]
    start, end = candidate["span"]
    return (
        field in {"source", "translation"}
        and isinstance(item.get(field), str)
        and 0 <= start < end <= len(item[field])
        and evidence == anchor(item[field], field, start, end)
    )
