"""Compact DOM snapshots.

A full DOM is far too large for a 7B local model, and most of it is irrelevant
to finding a control. This module extracts only interactive/labelled elements
with the attributes that carry identity, plus a generated unique CSS path used
as the last-resort selector when healing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

# Executed in the page. Kept dependency-free and defensive: a broken snapshot
# must never take the test down with it.
SNAPSHOT_JS = r"""
(limit) => {
  const SELECTOR = 'a,button,input,select,textarea,summary,[role],[data-testid],[data-test],[data-qa],[contenteditable="true"],label,h1,h2,h3,[aria-label]';
  const uniquePath = (el) => {
    if (el.id && document.querySelectorAll(CSS.escape ? '#' + CSS.escape(el.id) : '#' + el.id).length === 1) {
      return '#' + (CSS.escape ? CSS.escape(el.id) : el.id);
    }
    const parts = [];
    let node = el;
    while (node && node.nodeType === 1 && parts.length < 6) {
      let part = node.tagName.toLowerCase();
      if (node.id) { parts.unshift(part + '#' + node.id); break; }
      const parent = node.parentElement;
      if (parent) {
        const siblings = Array.from(parent.children).filter(c => c.tagName === node.tagName);
        if (siblings.length > 1) part += ':nth-of-type(' + (siblings.indexOf(node) + 1) + ')';
      }
      parts.unshift(part);
      node = node.parentElement;
    }
    return parts.join(' > ');
  };
  const labelFor = (el) => {
    if (el.labels && el.labels.length) return (el.labels[0].innerText || '').trim();
    const wrapper = el.closest('label');
    if (wrapper) return (wrapper.innerText || '').trim();
    const id = el.getAttribute('id');
    if (id) {
      const lbl = document.querySelector('label[for="' + id + '"]');
      if (lbl) return (lbl.innerText || '').trim();
    }
    return '';
  };
  const out = [];
  const seen = new Set();
  for (const el of document.querySelectorAll(SELECTOR)) {
    if (out.length >= limit) break;
    if (seen.has(el)) continue;
    seen.add(el);
    const rect = el.getBoundingClientRect();
    const style = window.getComputedStyle(el);
    const visible = rect.width > 0 && rect.height > 0 &&
      style.visibility !== 'hidden' && style.display !== 'none' && style.opacity !== '0';
    const text = (el.innerText || el.value || '').trim().slice(0, 80);
    out.push({
      tag: el.tagName.toLowerCase(),
      id: el.getAttribute('id') || '',
      name: el.getAttribute('name') || '',
      type: el.getAttribute('type') || '',
      testid: el.getAttribute('data-testid') || el.getAttribute('data-test') || el.getAttribute('data-qa') || '',
      aria: el.getAttribute('aria-label') || '',
      role: el.getAttribute('role') || '',
      placeholder: el.getAttribute('placeholder') || '',
      title: el.getAttribute('title') || '',
      cls: (el.getAttribute('class') || '').slice(0, 80),
      label: labelFor(el),
      text: text,
      visible: visible,
      editable: !!el.isContentEditable,
      css: uniquePath(el)
    });
  }
  return out;
}
"""

# Executed against one already-located element. Returns just the facts needed to
# decide whether it is the *kind* of element a description names ("log in
# button" must not bind to an <h1> that merely contains the words).
ELEMENT_INFO_JS = r"""
(el) => ({
  tag: el.tagName.toLowerCase(),
  type: (el.getAttribute('type') || '').toLowerCase(),
  role: (el.getAttribute('role') || '').toLowerCase(),
  href: el.hasAttribute('href'),
  editable: !!el.isContentEditable,
  text: ((el.innerText || el.value || el.getAttribute('aria-label') || '') + '').trim().slice(0, 160)
})
"""


@dataclass
class ElementInfo:
    tag: str = ""
    id: str = ""
    name: str = ""
    type: str = ""
    testid: str = ""
    aria: str = ""
    role: str = ""
    placeholder: str = ""
    title: str = ""
    cls: str = ""
    label: str = ""
    text: str = ""
    visible: bool = True
    editable: bool = False
    css: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ElementInfo:
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def identity_text(self) -> str:
        """Everything a human would read to name this element."""
        return " ".join(
            part for part in (
                self.testid, self.id, self.name, self.aria, self.label,
                self.placeholder, self.title, self.text, self.type, self.role, self.tag,
            ) if part
        )

    def line(self) -> str:
        bits = [f"<{self.tag}"]
        for key in ("id", "name", "type", "testid", "aria", "role", "placeholder", "label"):
            value = getattr(self, key)
            if value:
                bits.append(f'{key}="{value}"')
        bits.append(">")
        line = " ".join(bits)
        if self.text:
            line += f" {self.text}"
        return line + (f"  [css: {self.css}]" if self.css else "")


def capture(page: Any, limit: int = 120, visible_only: bool = True) -> list[ElementInfo]:
    try:
        raw = page.evaluate(SNAPSHOT_JS, limit)
    except Exception:
        return []
    items = [ElementInfo.from_dict(r) for r in raw or []]
    return [i for i in items if i.visible] if visible_only else items


def inspect_locator(locator: Any, timeout_ms: int = 500) -> dict[str, Any] | None:
    """Tag/type/role facts for the first element *locator* points at.

    Returns ``None`` when the element cannot be inspected (detached mid-call, a
    locator type without ``evaluate``). Callers treat that as "unknown", never as
    a match or a mismatch, so a flaky inspection cannot fail a test on its own.
    """
    try:
        try:
            info = locator.evaluate(ELEMENT_INFO_JS, timeout=timeout_ms)
        except TypeError:  # a harness locator that takes no timeout keyword
            info = locator.evaluate(ELEMENT_INFO_JS)
    except Exception:
        return None
    return info if isinstance(info, dict) else None


def render(elements: Sequence[ElementInfo], limit: int = 60) -> str:
    return "\n".join(f"{i}. {el.line()}" for i, el in enumerate(elements[:limit]))
