"""
Phase 6a — Gmail integration (read-only).

Lets Jarvis search/read your Gmail inbox as a tool, the same pattern as
every other tool in this project: the model decides when to call it, we
run the actual API call, the result goes back to the model as plain text.

READ-ONLY by design (Gmail scope: gmail.readonly). This tool cannot send,
delete, modify, or mark anything as read — it can only look. That's a
deliberate choice: a 3B local model occasionally over-triggers tool calls
(a known, accepted limitation throughout this project), which is harmless
for a lookup but would NOT be harmless for a send/delete action.

Setup is a one-time OAuth dance, unlike Ollama/RAG/memory which need
nothing: Google requires you to register your own "app" (free, takes a
few minutes) in Google Cloud Console and download a credentials.json file.
The first time this tool actually runs, it opens your browser for you to
approve access to your own Gmail account, then saves a refresh token to
disk so you never have to approve again. Full steps are in the README.

Two files this creates, BOTH already covered by .gitignore's `data/*`
pattern — neither should ever reach GitHub:
  - credentials.json  (your app's identity — copy this into the project
    root yourself, see README)
  - data/gmail_token.json  (your personal access/refresh token — created
    automatically after the first approval, not something you create)
"""

import os

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Read-only scope — deliberately the narrowest Gmail scope that exists.
# Changing this to a write scope requires deleting data/gmail_token.json
# and re-approving, since a token is only valid for the scope(s) it was
# originally granted.
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

CREDENTIALS_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "credentials.json"
)
TOKEN_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "gmail_token.json"
)

_service = None


def _get_gmail_service():
    global _service
    if _service is not None:
        return _service

    if not os.path.exists(CREDENTIALS_FILE):
        raise FileNotFoundError(
            "credentials.json not found in the project root. Download it from "
            "Google Cloud Console (see README's Gmail setup steps) and place "
            "it there before using email tools."
        )

    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # First-time setup only: opens your default browser to approve
            # access, then Google redirects to a temporary local server
            # this spins up on a random free port to catch the response.
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
        with open(TOKEN_FILE, "w", encoding="utf-8") as token_file:
            token_file.write(creds.to_json())

    _service = build("gmail", "v1", credentials=creds)
    return _service


def search_emails(query: str, max_results: int = 5) -> str:
    """Searches Gmail using Gmail's own search syntax (e.g. 'is:unread',
    'from:someone@example.com', 'subject:invoice', 'newer_than:2d') and
    returns sender, subject, date, and a short snippet for each match.
    Read-only — never modifies, sends, or deletes anything."""
    try:
        service = _get_gmail_service()
    except FileNotFoundError as exc:
        return str(exc)

    try:
        max_results = max(1, min(int(max_results), 10))  # sane bounds
        results = (
            service.users()
            .messages()
            .list(userId="me", q=query, maxResults=max_results)
            .execute()
        )
    except Exception as exc:  # noqa: BLE001 — surface a readable error to the model
        return f"Gmail search failed: {exc}"

    message_refs = results.get("messages", [])
    if not message_refs:
        return f"No emails found matching '{query}'."

    summaries = []
    for ref in message_refs:
        msg = (
            service.users()
            .messages()
            .get(
                userId="me",
                id=ref["id"],
                format="metadata",
                metadataHeaders=["From", "Subject", "Date"],
            )
            .execute()
        )
        headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
        snippet = msg.get("snippet", "")
        summaries.append(
            f"From: {headers.get('From', '(unknown)')}\n"
            f"Subject: {headers.get('Subject', '(no subject)')}\n"
            f"Date: {headers.get('Date', '(unknown)')}\n"
            f"Snippet: {snippet}"
        )

    return "\n\n---\n\n".join(summaries)