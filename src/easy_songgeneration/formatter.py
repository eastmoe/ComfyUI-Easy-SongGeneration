from __future__ import annotations

import re

INSTRUMENTAL_TAGS = (
    "intro-short",
    "intro-medium",
    "inst-short",
    "inst-medium",
    "outro-short",
    "outro-medium",
)
LYRIC_TAGS = ("verse", "chorus", "bridge")
ALL_SECTION_TAGS = INSTRUMENTAL_TAGS + LYRIC_TAGS
SECTION_CHOICES = ("none",) + LYRIC_TAGS
LANGUAGE_CHOICES = ("auto", "zh", "en")


def _normalize_section_tag(value: str) -> str:
    tag = (value or "").strip().lower()
    return tag if tag in ALL_SECTION_TAGS else ""


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _normalize_lyric_punctuation(text: str) -> str:
    replacements = {
        "。": ".",
        "．": ".",
        "！": ".",
        "？": ".",
        "，": ".",
        "、": ".",
        "；": ";",
        "：": ":",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "\u3000": " ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _clean_lyric_body(text: str, language: str) -> str:
    text = _normalize_lyric_punctuation(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for separator in (";", "!", "?"):
        text = text.replace(separator, ".")
    parts = []
    for piece in text.replace("\n", ".").split("."):
        cleaned = " ".join(piece.strip().strip(",").split())
        if cleaned:
            parts.append(cleaned)
    if not parts:
        return ""
    body = ".".join(parts)
    if language == "en" and not body.endswith("."):
        body += "."
    return body


def format_lyrics_text(raw: str, language: str, wrap_untagged_as: str) -> str:
    text = _normalize_lyric_punctuation(raw or "").strip()
    if not text:
        return ""

    tag_pattern = re.compile(
        r"\[(intro-short|intro-medium|inst-short|inst-medium|outro-short|outro-medium|verse|chorus|bridge)\]",
        re.IGNORECASE,
    )
    matches = list(tag_pattern.finditer(text))
    resolved_language = language if language in {"zh", "en"} else ("zh" if _contains_cjk(text) else "en")

    if not matches:
        tag = _normalize_section_tag(wrap_untagged_as)
        body = _clean_lyric_body(text, resolved_language)
        if tag and body:
            return f"[{tag}] {body}"
        return body

    sections = []
    for index, match in enumerate(matches):
        tag = _normalize_section_tag(match.group(1))
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : next_start].strip().strip(";")
        if tag in INSTRUMENTAL_TAGS:
            sections.append(f"[{tag}]")
            continue
        formatted_body = _clean_lyric_body(body, resolved_language)
        if formatted_body:
            sections.append(f"[{tag}] {formatted_body}")

    return " ; ".join(sections)


def format_style_text(raw: str, trailing_period: bool) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    for source, target in {
        "，": ",",
        "、": ",",
        "；": ",",
        ";": ",",
        "\n": ",",
        "\r": ",",
        "\u3000": " ",
    }.items():
        text = text.replace(source, target)
    tags = []
    seen = set()
    for part in text.split(","):
        tag = " ".join(part.strip().strip(".").split())
        key = tag.lower()
        if tag and key not in seen:
            seen.add(key)
            tags.append(tag)
    result = ", ".join(tags)
    if trailing_period and result and not result.endswith("."):
        result += "."
    return result
