"""Prompt templates, answer extraction and scoring shared by inference and analysis."""
import re
import string

DIRECT_SUFFIX = ('Give only the final answer, with no explanation or reasoning. '
                 'Respond in the format "Answer: <answer>".')
COT_SUFFIX = ('Think step by step and write out your reasoning. '
              'Then finish with a final line in the format "Answer: <answer>".')
# Paraphrased reasoning instruction used for the prompt-robustness check.
COT2_SUFFIX = ('Before answering, carefully work through the problem and explain your thought process. '
               'When you are done, end your response with one last line of the form "Answer: <answer>".')

CONDITIONS = {
    # name: (prompt suffix, assistant prefill, max_new_tokens)
    "direct": (DIRECT_SUFFIX, "Answer:", 24),
    "cot": (COT_SUFFIX, "", 1024),
    "cot2": (COT2_SUFFIX, "", 1024),
}


def build_user_prompt(item, condition):
    return f"{item['text']}\n\n{CONDITIONS[condition][0]}"


# "Answer: X", "answer is X", "answer is: X" all count as an explicit final-answer marker.
MARKER = re.compile(r"answer\s*:|answer\s+is\s*:?", flags=re.I)


def _clean(seg):
    """Strip markdown / LaTeX decoration around an answer segment."""
    seg = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", seg)
    seg = re.sub(r"\\text\{([^{}]*)\}", r"\1", seg)
    seg = re.sub(r"\\[()\[\]]|[*`$]", "", seg)
    return seg.strip()


def extract_answer(output, condition):
    """Return (answer_segment, has_marker).

    Direct outputs are a continuation of the prefilled "Answer:", so the first line is the answer.
    CoT outputs: the first non-empty line after the LAST final-answer marker; if there is no marker,
    fall back to the last non-empty line and flag the response (has_marker=False).
    """
    out = output.strip()
    if condition == "direct":
        return _clean(out.split("\n")[0]), True
    hits = list(MARKER.finditer(out))
    if hits:
        lines = [_clean(l) for l in out[hits[-1].end():].split("\n")]
        lines = [l for l in lines if l]
        return (lines[0] if lines else ""), True
    lines = [_clean(l) for l in out.split("\n")]
    lines = [l for l in lines if l]
    return (lines[-1] if lines else ""), False


def _norm(s):
    """TriviaQA-style normalisation: lowercase, drop punctuation/articles, squeeze whitespace."""
    s = s.lower()
    s = "".join(ch if ch not in set(string.punctuation + "‘’´`_") else " " for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _last_number(s):
    s = s.replace(",", "").replace("$", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(nums[-1]) if nums else None


def score(item, answer):
    """Return 1 if the extracted answer segment is correct for this item, else 0."""
    t = item["answer_type"]
    a = answer.replace("*", "").strip()
    if t == "label":
        low = a.lower()
        # spelling variants of the AG News label ("Science/Tech", "Sci-Tech", "Science and Technology")
        low = re.sub(r"sci(?:ence)?\s*(?:/|-|&|and)\s*tech(?:nology)?", "sci/tech", low)
        # exactly one of the allowed labels may appear (word-boundary match), and it must be the gold one
        found = [l for l in item["labels"] if re.search(rf"(?<![a-z]){re.escape(l.lower())}(?![a-z])", low)]
        return int(found == [item["gold"]])
    if t == "mcq":
        letters = "ABCDE"[: item["n_options"]]
        m = re.match(rf"^\(?([{letters}])\)?(?![A-Za-z])", a)
        if m:
            return int(m.group(1) == item["gold"])
        # otherwise accept a single unambiguous parenthesised option letter, e.g. "option (C)"
        par = set(re.findall(rf"\(([{letters}])\)", a))
        return int(par == {item["gold"]})
    if t == "number":
        v = _last_number(a)
        return int(v is not None and abs(v - float(item["gold"])) < 1e-4)
    if t == "freeform":
        na = _norm(a)
        return int(any(al and re.search(rf"(?<!\w){re.escape(al)}(?!\w)", na) for al in item["gold"]))
    raise ValueError(t)
