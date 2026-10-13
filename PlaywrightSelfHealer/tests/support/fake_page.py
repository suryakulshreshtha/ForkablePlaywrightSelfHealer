"""A miniature headless browser for tests.

Playwright's browser bundles are a ~150 MB download from a CDN. That is fine on
a laptop and impossible inside a locked-down build box, so the resolver,
healer, memory and executor would otherwise go untested exactly where it matters
most.

``FakePage`` closes that gap. It speaks HTTP to the real bundled demo app over
loopback, parses the real HTML, and implements the slice of the Playwright page
API the agent actually calls: ``locator``, ``get_by_role``, ``get_by_label``,
``get_by_placeholder``, ``get_by_text``, ``evaluate`` (returning the same
snapshot shape as the injected JS), ``goto``, ``inner_text`` and ``title``.

It is not a browser - no CSS, no layout, no JavaScript. It is a faithful enough
DOM to prove that a description binds to the element a human would have picked,
which is the entire claim self-healing makes.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from html.parser import HTMLParser
from typing import Any

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr"}

_ROLE_BY_TAG = {
    "button": "button", "select": "combobox", "textarea": "textbox",
    "h1": "heading", "h2": "heading", "h3": "heading", "form": "form",
    "table": "table", "ul": "list", "li": "listitem",
}
_ROLE_BY_INPUT_TYPE = {
    "text": "textbox", "password": "textbox", "email": "textbox",
    "search": "searchbox", "tel": "textbox", "url": "textbox", "number": "spinbutton",
    "checkbox": "checkbox", "radio": "radio", "submit": "button", "button": "button",
}


class Node:
    __slots__ = ("tag", "attrs", "children", "parent", "own_text", "value")

    def __init__(self, tag: str, attrs: dict[str, str], parent: Node | None = None) -> None:
        self.tag = tag
        self.attrs = attrs
        self.children: list[Node] = []
        self.parent = parent
        self.own_text = ""
        self.value = attrs.get("value", "")

    # -- traversal -----------------------------------------------------
    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()

    @property
    def text(self) -> str:
        parts = [self.own_text] + [c.text for c in self.children]
        return re.sub(r"\s+", " ", " ".join(p for p in parts if p)).strip()

    def get(self, name: str, default: str = "") -> str:
        return self.attrs.get(name, default)

    @property
    def visible(self) -> bool:
        if self.tag == "input" and self.get("type") == "hidden":
            return False
        node: Node | None = self
        while node is not None:
            if node.tag in {"head", "script", "style", "title"}:
                return False
            if node.get("hidden") is not None and "hidden" in node.attrs:
                return False
            node = node.parent
        return True

    @property
    def css_path(self) -> str:
        if self.get("id"):
            return f"#{self.get('id')}"
        parts: list[str] = []
        node: Node | None = self
        while node is not None and node.tag not in {"html", "[document]"} and len(parts) < 6:
            part = node.tag
            if node.get("id"):
                parts.insert(0, f"{node.tag}#{node.get('id')}")
                break
            parent = node.parent
            if parent is not None:
                siblings = [c for c in parent.children if c.tag == node.tag]
                if len(siblings) > 1:
                    part += f":nth-of-type({siblings.index(node) + 1})"
            parts.insert(0, part)
            node = parent
        return " > ".join(parts)

    # -- semantics -----------------------------------------------------
    @property
    def role(self) -> str:
        explicit = self.get("role")
        if explicit:
            return explicit
        if self.tag == "input":
            return _ROLE_BY_INPUT_TYPE.get(self.get("type", "text"), "textbox")
        if self.tag == "a":
            return "link" if self.get("href") else ""
        return _ROLE_BY_TAG.get(self.tag, "")

    def label_text(self, document: Document) -> str:
        node_id = self.get("id")
        if node_id:
            for candidate in document.root.walk():
                if candidate.tag == "label" and candidate.get("for") == node_id:
                    return candidate.text
        node = self.parent
        while node is not None:
            if node.tag == "label":
                return node.text
            node = node.parent
        return ""

    def accessible_name(self, document: Document) -> str:
        return (
            self.get("aria-label")
            or self.label_text(document)
            or (self.text if self.tag not in {"input", "select", "textarea"} else "")
            or self.get("placeholder")
            or self.get("title")
        ).strip()


class _Builder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node("[document]", {})
        self.current = self.root

    def handle_starttag(self, tag: str, attrs: Sequence[tuple]) -> None:
        node = Node(tag, {k: (v if v is not None else "") for k, v in attrs}, self.current)
        self.current.children.append(node)
        if tag not in VOID:
            self.current = node

    def handle_startendtag(self, tag: str, attrs: Sequence[tuple]) -> None:
        node = Node(tag, {k: (v if v is not None else "") for k, v in attrs}, self.current)
        self.current.children.append(node)

    def handle_endtag(self, tag: str) -> None:
        node: Node | None = self.current
        while node is not None and node.tag != tag:
            node = node.parent
        if node is not None and node.parent is not None:
            self.current = node.parent

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.current.own_text += (" " if self.current.own_text else "") + data.strip()


class Document:
    def __init__(self, html: str) -> None:
        builder = _Builder()
        builder.feed(html)
        self.root = builder.root

    @property
    def title(self) -> str:
        for node in self.root.walk():
            if node.tag == "title":
                return node.text
        return ""

    @property
    def body_text(self) -> str:
        for node in self.root.walk():
            if node.tag == "body":
                return node.text
        return self.root.text

    def elements(self) -> list[Node]:
        return [n for n in self.root.walk() if n.tag not in {"[document]", "html", "head", "body"}]


# ----------------------------------------------------------------------
_ATTR_RE = re.compile(r'\[([\w-]+)([*^$]?=)"([^"]*)"(\s*i)?\]')
_SIMPLE_RE = re.compile(r'^([a-zA-Z]+)?(#[\w-]+)?((?:\[[^\]]+\])*)(?::nth-of-type\((\d+)\))?$')


def _matches_simple(node: Node, selector: str) -> bool:
    selector = selector.strip()
    match = _SIMPLE_RE.match(selector)
    if not match:
        return False
    tag, node_id, attrs, nth = match.groups()
    if tag and node.tag != tag.lower():
        return False
    if node_id and node.get("id") != node_id[1:]:
        return False
    for attr, op, value, ci in _ATTR_RE.findall(attrs or ""):
        actual = node.get(attr)
        if attr not in node.attrs:
            return False
        a, b = (actual.lower(), value.lower()) if ci else (actual, value)
        if op == "=" and a != b:
            return False
        if op == "*=" and b not in a:
            return False
        if op == "^=" and not a.startswith(b):
            return False
        if op == "$=" and not a.endswith(b):
            return False
    if nth and node.parent is not None:
        siblings = [c for c in node.parent.children if c.tag == node.tag]
        if siblings.index(node) + 1 != int(nth):
            return False
    return True


def query(document: Document, selector: str) -> list[Node]:
    """Support simple selectors plus ``a > b > c`` chains produced by healing."""
    parts = [p.strip() for p in selector.split(">")]
    candidates = [n for n in document.elements() if _matches_simple(n, parts[-1])]
    if len(parts) == 1:
        return candidates

    def chain_ok(node: Node) -> bool:
        current = node.parent
        for part in reversed(parts[:-1]):
            if current is None or not _matches_simple(current, part):
                return False
            current = current.parent
        return True

    return [n for n in candidates if chain_ok(n)]


# ----------------------------------------------------------------------
class FakeLocator:
    def __init__(self, page: FakePage, nodes: Sequence[Node], description: str = "") -> None:
        self.page = page
        self.nodes = list(nodes)
        self.description = description

    def count(self) -> int:
        return len(self.nodes)

    @property
    def first(self) -> FakeLocator:
        return FakeLocator(self.page, self.nodes[:1], self.description)

    def _node(self) -> Node:
        if not self.nodes:
            raise RuntimeError(f"could not locate {self.description!r}: no element matches")
        return self.nodes[0]

    def is_visible(self) -> bool:
        return bool(self.nodes) and self.nodes[0].visible

    def evaluate(self, script: str, arg: Any = None, timeout: int = 0) -> dict[str, Any]:
        """Mirror of ``ELEMENT_INFO_JS``: the kind-of-element facts for the first node."""
        node = self._node()
        editable = "contenteditable" in node.attrs and node.get("contenteditable") != "false"
        return {
            "tag": node.tag,
            "type": node.get("type").lower(),
            "role": node.get("role").lower(),
            "href": "href" in node.attrs,
            "editable": editable,
            "text": (node.text.strip() or node.value or node.get("aria-label")).strip()[:160],
        }

    def fill(self, value: str, timeout: int = 0) -> None:
        self._node().value = value

    def input_value(self, timeout: int = 0) -> str:
        return self._node().value

    def press(self, key: str, timeout: int = 0) -> None:
        if key.lower() == "enter":
            self.page.submit_form(self._node())

    def check(self, timeout: int = 0) -> None:
        self._node().attrs["checked"] = "checked"

    def uncheck(self, timeout: int = 0) -> None:
        self._node().attrs.pop("checked", None)

    def hover(self, timeout: int = 0) -> None:
        self._node()

    def select_option(self, value: str, timeout: int = 0) -> None:
        self._node().value = value

    def click(self, timeout: int = 0) -> None:
        node = self._node()
        if node.tag == "a" and node.get("href"):
            self.page.goto(urllib.parse.urljoin(self.page.url, node.get("href")))
            return
        if node.tag == "button" or (node.tag == "input" and node.get("type") in {"submit", "button"}):
            self.page.submit_form(node)
            return
        if node.tag in {"input"} and node.get("type") in {"checkbox", "radio"}:
            node.attrs["checked"] = "checked"


class FakeResponse:
    """Mirrors the slice of Playwright's APIResponse the agent uses."""

    def __init__(self, status: int, body: bytes, headers: Any) -> None:
        self.status = status
        self._body = body
        self.headers = headers

    def json(self) -> Any:
        import json as _json

        return _json.loads(self._body.decode("utf-8"))

    def text(self) -> str:
        return self._body.decode("utf-8", "replace")


class FakeRequestContext:
    """``page.request`` - shares the page's cookie jar, exactly like Playwright."""

    def __init__(self, page: FakePage) -> None:
        self.page = page

    def _call(self, method: str, url: str, data: Any = None, timeout: int = 0) -> FakeResponse:
        import json as _json

        body = None
        headers = {}
        if data is not None:
            body = data.encode() if isinstance(data, str) else _json.dumps(data).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=body, method=method.upper())
        for key, value in headers.items():
            request.add_header(key, value)
        if self.page.cookies:
            request.add_header(
                "Cookie", "; ".join(f"{k}={v}" for k, v in self.page.cookies.items())
            )
        try:
            response = urllib.request.urlopen(request, timeout=10)
            status, payload, resp_headers = response.status, response.read(), response.headers
        except urllib.error.HTTPError as exc:
            status, payload, resp_headers = exc.code, exc.read(), exc.headers
        for value in resp_headers.get_all("Set-Cookie") or []:
            name, _, rest = value.partition("=")
            self.page.cookies[name.strip()] = rest.split(";")[0]
        return FakeResponse(status, payload, resp_headers)

    def get(self, url: str, **kw: Any) -> FakeResponse:
        return self._call("GET", url, **kw)

    def post(self, url: str, **kw: Any) -> FakeResponse:
        return self._call("POST", url, **kw)

    def put(self, url: str, **kw: Any) -> FakeResponse:
        return self._call("PUT", url, **kw)

    def patch(self, url: str, **kw: Any) -> FakeResponse:
        return self._call("PATCH", url, **kw)

    def delete(self, url: str, **kw: Any) -> FakeResponse:
        return self._call("DELETE", url, **kw)

    def head(self, url: str, **kw: Any) -> FakeResponse:
        return self._call("HEAD", url, **kw)


class FakePage:
    """Playwright-shaped page backed by urllib and an HTML parser."""

    def __init__(self, base_url: str = "") -> None:
        self.base_url = base_url
        self.url = ""
        self.document = Document("<html><body></body></html>")
        self.cookies: dict[str, str] = {}
        self.history: list[str] = []
        self.request = FakeRequestContext(self)

    # -- navigation ----------------------------------------------------
    def goto(self, url: str, wait_until: str = "", **_: Any) -> None:
        target = urllib.parse.urljoin(self.base_url or self.url or "", url)
        self._request("GET", target)

    def submit_form(self, node: Node) -> None:
        form: Node | None = node
        while form is not None and form.tag != "form":
            form = form.parent
        if form is None:
            return
        data: dict[str, str] = {}
        for field in form.walk():
            if field.tag in {"input", "textarea", "select"} and field.get("name"):
                if field.get("type") in {"checkbox", "radio"} and "checked" not in field.attrs:
                    continue
                data[field.get("name")] = field.value
        action = urllib.parse.urljoin(self.url, form.get("action") or self.url)
        method = (form.get("method") or "get").upper()
        if method == "POST":
            self._request("POST", action, urllib.parse.urlencode(data).encode())
        else:
            self._request("GET", f"{action}?{urllib.parse.urlencode(data)}")

    def _request(self, method: str, url: str, body: bytes | None = None, depth: int = 0) -> None:
        if depth > 8:
            raise RuntimeError("too many redirects")
        request = urllib.request.Request(url, data=body, method=method)
        if self.cookies:
            request.add_header(
                "Cookie", "; ".join(f"{k}={v}" for k, v in self.cookies.items())
            )
        if body is not None:
            request.add_header("Content-Type", "application/x-www-form-urlencoded")

        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args: Any, **kwargs: Any) -> None:
                return None

        opener = urllib.request.build_opener(_NoRedirect)
        try:
            response = opener.open(request, timeout=10)
            status, headers, payload = response.status, response.headers, response.read()
        except urllib.error.HTTPError as exc:
            status, headers, payload = exc.code, exc.headers, exc.read()

        for value in headers.get_all("Set-Cookie") or []:
            name, _, rest = value.partition("=")
            self.cookies[name.strip()] = rest.split(";")[0]

        if status in {301, 302, 303, 307, 308}:
            location = headers.get("Location", "")
            self._request("GET", urllib.parse.urljoin(url, location), None, depth + 1)
            return

        self.url = url
        self.history.append(url)
        self.document = Document(payload.decode("utf-8", "replace"))

    # -- queries -------------------------------------------------------
    def locator(self, selector: str) -> FakeLocator:
        if selector.startswith("xpath="):
            return FakeLocator(self, [], selector)
        if selector == "body":
            body = [n for n in self.document.root.walk() if n.tag == "body"]
            return FakeLocator(self, body, selector)
        return FakeLocator(self, query(self.document, selector), selector)

    def get_by_role(self, role: str, name: str = "", exact: bool = False) -> FakeLocator:
        out: list[Node] = []
        for node in self.document.elements():
            if node.role != role:
                continue
            accessible = node.accessible_name(self.document)
            if not name:
                out.append(node)
            elif exact and accessible.strip().lower() == name.strip().lower():
                out.append(node)
            elif not exact and name.strip().lower() in accessible.lower():
                out.append(node)
        return FakeLocator(self, out, f"role={role}[name={name}]")

    def get_by_label(self, text: str, exact: bool = False) -> FakeLocator:
        out = []
        for node in self.document.elements():
            if node.tag not in {"input", "select", "textarea"}:
                continue
            label = node.label_text(self.document).strip().lower()
            needle = text.strip().lower()
            if (label == needle) if exact else (needle in label and label != ""):
                out.append(node)
        return FakeLocator(self, out, f"label={text}")

    def get_by_placeholder(self, text: str, exact: bool = False) -> FakeLocator:
        needle = text.strip().lower()
        out = [
            n for n in self.document.elements()
            if n.get("placeholder") and (
                n.get("placeholder").lower() == needle if exact else needle in n.get("placeholder").lower()
            )
        ]
        return FakeLocator(self, out, f"placeholder={text}")

    def get_by_text(self, text: str, exact: bool = False) -> FakeLocator:
        needle = text.strip().lower()
        out = []
        for node in self.document.elements():
            if node.tag in {"script", "style"}:
                continue
            content = node.text.strip().lower()
            if not content:
                continue
            if (content == needle) if exact else (needle in content):
                if not any(needle in c.text.strip().lower() for c in node.children if c.text):
                    out.append(node)
        return FakeLocator(self, out, f"text={text}")

    # -- page-level ----------------------------------------------------
    def inner_text(self, selector: str) -> str:
        return self.document.body_text

    def content(self) -> str:
        return self.document.body_text

    def title(self) -> str:
        return self.document.title

    def screenshot(self, path: str = "", full_page: bool = False, **_: Any) -> bytes:
        return b""

    def evaluate(self, script: str, arg: Any = None) -> list[dict[str, Any]]:
        """Return the same shape the injected snapshot JS produces."""
        if isinstance(arg, dict):  # e.g. the DOM-quiet wait: nothing mutates here
            return None  # type: ignore[return-value]
        limit = int(arg or 120)
        out: list[dict[str, Any]] = []
        interesting = {"a", "button", "input", "select", "textarea", "label", "h1", "h2", "h3"}
        for node in self.document.elements():
            if len(out) >= limit:
                break
            if node.tag not in interesting and not node.get("role") \
                    and not node.get("data-testid") and not node.get("aria-label"):
                continue
            out.append({
                "tag": node.tag,
                "id": node.get("id"),
                "name": node.get("name"),
                "type": node.get("type"),
                "testid": node.get("data-testid") or node.get("data-test") or node.get("data-qa"),
                "aria": node.get("aria-label"),
                "role": node.get("role"),
                "placeholder": node.get("placeholder"),
                "title": node.get("title"),
                "cls": node.get("class")[:80],
                "label": node.label_text(self.document),
                "text": node.text[:80],
                "visible": node.visible,
                "editable": "contenteditable" in node.attrs and node.get("contenteditable") != "false",
                "css": node.css_path,
            })
        return out


class FakeSession:
    """Stands in for :class:`BrowserSession` in tests."""

    def __init__(self, base_url: str) -> None:
        self.page = FakePage(base_url)
        self.shots: list[str] = []

    def screenshot(self, name: str, full_page: bool = False) -> str:
        self.shots.append(name)
        return ""

    def __enter__(self) -> FakeSession:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def page_from_html(html: str, url: str = "") -> FakePage:
    """A FakePage holding ``html`` with no server behind it (for unit tests)."""
    page = FakePage()
    page.url = url
    page.document = Document(f"<html><body>{html}</body></html>")
    return page
