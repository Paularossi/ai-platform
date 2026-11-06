"""Render provider search citations as portable links; keep raw metadata in call logs."""
from urllib.parse import quote, urlsplit


def _link(url, label="source"):
    if not isinstance(url, str) or urlsplit(url).scheme not in ("http", "https"):
        return ""
    return "[" + label + "](<" + quote(url, safe=":/?&=%#@+;,$!~*'") + ">)"


def openai_text(output):
    parts = []
    for item in output:
        if item.get("type") != "message":
            continue
        for block in item.get("content", []):
            if block.get("type") != "output_text":
                continue
            text = block.get("text", "")
            for citation in sorted(block.get("annotations", []), key=lambda c: c.get("start_index", 0), reverse=True):
                link = _link(citation.get("url"))
                start, end = citation.get("start_index"), citation.get("end_index")
                if link and isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(text):
                    text = text[:start] + link + text[end:]
            parts.append(text)
    return "\n".join(parts)


def anthropic_text(blocks):
    parts = []
    for block in blocks:
        if block.get("type") == "text":
            links = [_link(c.get("url")) for c in block.get("citations") or []]
            parts.append(block.get("text", "") + " ".join(" " + link for link in links if link))
    return "".join(parts)


def google_text(text, metadata):
    # Gemini segment offsets are byte offsets, so splice UTF-8 rather than characters.
    data = text.encode("utf-8")
    chunks = metadata.get("grounding_chunks") or []
    inserts = {}
    for support in metadata.get("grounding_supports") or []:
        end = (support.get("segment") or {}).get("end_index")
        if not isinstance(end, int) or not 0 <= end <= len(data):
            continue
        for index in support.get("grounding_chunk_indices") or []:
            if isinstance(index, int) and 0 <= index < len(chunks):
                link = _link((chunks[index].get("web") or {}).get("uri"), f"source {index + 1}")
                if link and link not in inserts.setdefault(end, []):
                    inserts[end].append(link)
    for end, links in sorted(inserts.items(), reverse=True):
        data = data[:end] + (" " + " ".join(links)).encode("utf-8") + data[end:]
    return data.decode("utf-8")


def search_suggestions_frame(search):
    """Isolate Google's provided Search Suggestions HTML from the parent page."""
    from html import escape
    markup = ((search.get("grounding_metadata") or {}).get("search_entry_point") or {}).get("rendered_content")
    if not markup:
        return ""
    return ('<iframe title="Google Search suggestions" sandbox="allow-popups allow-popups-to-escape-sandbox" '
            'style="width:100%;height:160px;border:0" srcdoc="' + escape(markup, quote=True) + '"></iframe>')


def search_summary(agents, searches, enabled=True):
    """Provider-reported counts, retaining unknown counts and original query order."""
    rows = []
    for agent in agents:
        search = searches.get(agent, {})
        queries = []
        for item in search.get("output", []):
            action = item.get("action") or {}
            if action.get("type") == "search":
                queries.extend(action.get("queries") or ([action["query"]] if action.get("query") else []))
        for block in search.get("content", []):
            if block.get("type") == "server_tool_use" and block.get("name") == "web_search":
                query = (block.get("input") or {}).get("query")
                if query:
                    queries.append(query)
        queries.extend((search.get("grounding_metadata") or {}).get("web_search_queries") or [])
        count = search.get("query_count") if "query_count" in search else search.get("search_calls")
        rows.append({"Agent": agent, "Count": count if count is not None else ("Not reported" if enabled else "Disabled"),
                     "Unit": "Queries" if "query_count" in search else "Search calls", "Queries": queries})
    return rows
