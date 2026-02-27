"""
Route label constants for the supervisor graph.

Import these instead of hard-coding strings so that typos
become import errors rather than silent mis-routes.
"""

ROUTE_LOCAL_RAG = "local_rag"
ROUTE_WEB_RESEARCH = "web_research"
ROUTE_TIMELINE = "timeline_research"
ROUTE_MATH = "math"
ROUTE_AMBIGUOUS = "ambiguous"

# Every label the dispatcher will accept (excludes AMBIGUOUS)
VALID_ROUTES: list[str] = [
    ROUTE_LOCAL_RAG,
    ROUTE_WEB_RESEARCH,
    ROUTE_TIMELINE,
    ROUTE_MATH,
]

# Fallback when nothing else matches
DEFAULT_ROUTE = ROUTE_LOCAL_RAG
