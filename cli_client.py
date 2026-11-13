#!/usr/bin/env python3
"""
AMG Multimodal RAG — Interactive & Scriptable API CLI Debugger / Test Client

Features:
  - Persistent authentication: Login once, and the JWT token is saved to local session.
    All subsequent requests automatically attach `Authorization: Bearer <token>`.
  - Interactive mode selection for all endpoints:
      1.  GET  /health
      2.  POST /create_user
      3.  POST /login (auto-saves JWT & fetches profile)
      4.  GET  /me
      5.  GET  /list_documents
      6.  POST /ingest_document (with auto-polling option & PDF selector)
      7.  GET  /ingest_document/{job_id}
      8.  POST /query (with auto-polling option)
      9.  GET  /query/{query_job_id}
      10. POST /generate (with auto-polling option)
      11. GET  /generate/{generate_job_id}
      12. Automated Smoke Test (health -> register/login -> me -> list docs)
      13. Session / Token Management (inspect, switch, logout, base URL)
      14. Custom Raw Request (any HTTP method, path, and payload)
  - Rich debug visualization:
      - Request inspector: method, endpoint URL, authorization status, formatted JSON payload.
      - Response inspector: colored status badge, round-trip latency, response headers.
      - Formatted domain panels: parsed documents, retrieval candidates with scores/boxes/claims,
        markdown answer rendering, job progress polling, and server error diagnostics.
  - Scriptable non-interactive CLI flags for automation (e.g. --smoke-test, --endpoint, --poll).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

# ============================================================================
# Configuration & Constants
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SESSION_FILE = BASE_DIR / "data" / ".amg_session.json"
SESSION_FILE = Path(os.getenv("AMG_SESSION_FILE", str(DEFAULT_SESSION_FILE)))
DEFAULT_API_URL = os.getenv("AMG_API_URL", "http://127.0.0.1:8000").rstrip("/")

console = Console()


# ============================================================================
# Session & Token Persistence
# ============================================================================

class SessionManager:
    """Manages persistent credentials, JWT tokens, and CLI state across executions."""

    def __init__(self, session_path: Path = SESSION_FILE):
        self.session_path = session_path
        self.base_url: str = DEFAULT_API_URL
        self.access_token: Optional[str] = None
        self.user: dict[str, Any] = {}
        self.auto_auth: bool = True
        self.last_jobs: dict[str, str] = {
            "ingest": "",
            "query": "",
            "generate": "",
        }
        self.active_conversation_id: str = ""
        self.load()

    def load(self) -> None:
        """Load session state from disk if it exists."""
        if not self.session_path.exists():
            return
        try:
            with open(self.session_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.base_url = data.get("base_url", DEFAULT_API_URL).rstrip("/")
            self.access_token = data.get("access_token")
            self.user = data.get("user", {})
            self.auto_auth = data.get("auto_auth", True)
            self.last_jobs = data.get("last_jobs", self.last_jobs)
            self.active_conversation_id = data.get("active_conversation_id", "")
        except Exception as exc:
            console.print(f"[yellow]Warning: Could not read session file: {exc}[/yellow]")

    def save(self) -> None:
        """Persist current session state to disk."""
        try:
            self.session_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "base_url": self.base_url,
                "access_token": self.access_token,
                "user": self.user,
                "auto_auth": self.auto_auth,
                "last_jobs": self.last_jobs,
                "active_conversation_id": self.active_conversation_id,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            with open(self.session_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as exc:
            console.print(f"[yellow]Warning: Could not save session: {exc}[/yellow]")

    def clear(self) -> None:
        """Clear authenticated token and user profile."""
        self.access_token = None
        self.user = {}
        self.save()

    def set_login_data(self, access_token: str, user_dict: Optional[dict[str, Any]] = None) -> None:
        """Store new authentication tokens."""
        self.access_token = access_token.strip()
        if user_dict:
            self.user = user_dict
        self.save()

    def record_job(self, job_type: str, job_id: str) -> None:
        """Cache the most recent job ID for convenient auto-fill."""
        self.last_jobs[job_type] = job_id.strip()
        self.save()

    def is_authenticated(self) -> bool:
        return bool(self.access_token and self.access_token.strip())

    def get_auth_headers(self) -> dict[str, str]:
        if self.auto_auth and self.is_authenticated():
            return {"Authorization": f"Bearer {self.access_token}"}
        return {}

    def decode_token_unverified(self) -> Optional[dict[str, Any]]:
        """Decode JWT claims without verification for inspection in the CLI."""
        if not self.access_token:
            return None
        parts = self.access_token.split(".")
        if len(parts) < 2:
            return None
        try:
            padded = parts[1] + "=" * (-len(parts[1]) % 4)
            decoded = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
            return json.loads(decoded)
        except Exception:
            return None


# ============================================================================
# API Client & Debug Visualizer
# ============================================================================

class ApiDebugger:
    """HTTP client wrapper with rich debug inspection and specialized response formatters."""

    def __init__(self, session: SessionManager):
        self.session = session
        self.http = requests.Session()

    def send_request(
        self,
        method: str,
        path: str,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
        custom_headers: Optional[dict[str, str]] = None,
        use_auth: bool = True,
        show_debug: bool = True,
    ) -> requests.Response:
        """Send HTTP request, record round-trip latency, and format debug output."""
        method = method.upper().strip()
        url = f"{self.session.base_url}/{path.lstrip('/')}"

        headers = {
            "Accept": "application/json",
            "User-Agent": "AMG-CLI-Debugger/1.0",
        }
        if use_auth:
            headers.update(self.session.get_auth_headers())
        if custom_headers:
            headers.update(custom_headers)

        if show_debug:
            self._print_request_panel(method, url, headers, params, json_body)

        start_time = time.perf_counter()
        try:
            response = self.http.request(
                method=method,
                url=url,
                params=params,
                json=json_body,
                headers=headers,
                timeout=120,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        except requests.exceptions.ConnectionError as exc:
            console.print(
                Panel(
                    f"[bold red]Connection Failed:[/bold red] Could not connect to API at [cyan]{self.session.base_url}[/cyan].\n"
                    f"Is the FastAPI server running? (Start with: [bold green]uvicorn api:app --host 0.0.0.0 --port 8000[/bold green])\n"
                    f"[dim]Details: {exc}[/dim]",
                    title="⚠️ Network Connection Error",
                    border_style="red",
                )
            )
            return None
        except Exception as exc:
            console.print(
                Panel(
                    f"[bold red]Request Error:[/bold red] {exc}",
                    title="⚠️ Unexpected Exception",
                    border_style="red",
                )
            )
            return None

        if show_debug:
            self._print_response_panel(response, elapsed_ms)

        return response

    def _print_request_panel(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        params: Optional[dict[str, Any]],
        json_body: Optional[dict[str, Any]],
    ) -> None:
        """Format the outgoing HTTP request for debugging."""
        header_text = Text()
        for k, v in headers.items():
            if k.lower() == "authorization" and v.startswith("Bearer "):
                token = v[7:]
                preview = f"{token[:14]}...{token[-10:]} (len={len(token)})" if len(token) > 30 else token
                header_text.append(f"  {k}: ", style="bold cyan")
                header_text.append(f"Bearer {preview}\n", style="green")
            else:
                header_text.append(f"  {k}: ", style="bold cyan")
                header_text.append(f"{v}\n", style="white")

        body_renderable: Any = "[dim](empty payload)[/dim]"
        if json_body is not None:
            json_str = json.dumps(json_body, indent=2)
            body_renderable = Syntax(json_str, "json", theme="monokai", word_wrap=True)

        info_table = Table.grid(padding=(0, 1))
        info_table.add_column(style="bold yellow")
        info_table.add_column()

        info_table.add_row("Method:", f"[bold green]{method}[/bold green]")
        info_table.add_row("URL:", f"[cyan]{url}[/cyan]")
        if params:
            info_table.add_row("Params:", json.dumps(params))

        req_table = Table.grid(expand=True)
        req_table.add_row(info_table)
        req_table.add_row("\n[bold magenta]Headers:[/bold magenta]")
        req_table.add_row(header_text)
        if json_body is not None:
            req_table.add_row("[bold magenta]JSON Body:[/bold magenta]")
            req_table.add_row(body_renderable)

        console.print(
            Panel(
                req_table,
                title=f"[bold]🚀 HTTP Request: {method} {url}[/bold]",
                border_style="blue",
            )
        )

    def _print_response_panel(self, response: requests.Response, elapsed_ms: float) -> None:
        """Format the incoming HTTP response with syntax-highlighted body and visual badges."""
        code = response.status_code
        if 200 <= code < 300:
            badge_style = "bold green on dark_green"
            border_style = "green"
            badge_text = f" {code} {response.reason} "
        elif 300 <= code < 400:
            badge_style = "bold yellow on dark_yellow"
            border_style = "yellow"
            badge_text = f" {code} {response.reason} "
        elif 400 <= code < 500:
            badge_style = "bold white on dark_red"
            border_style = "red"
            badge_text = f" {code} {response.reason} "
        else:
            badge_style = "bold white on red"
            border_style = "bright_red"
            badge_text = f" {code} {response.reason} "

        meta_line = (
            f"[{badge_style}]{badge_text}[/{badge_style}]  "
            f"[bold cyan]Latency:[/bold cyan] {elapsed_ms:.1f} ms  "
            f"[bold cyan]Bytes:[/bold cyan] {len(response.content)}  "
            f"[bold cyan]Type:[/bold cyan] {response.headers.get('content-type', 'unknown')}"
        )

        raw_text = response.text
        is_json = False
        parsed_json = None
        try:
            parsed_json = response.json()
            is_json = True
        except Exception:
            pass

        console.print(
            Panel(
                meta_line,
                title="📬 HTTP Response Summary",
                border_style=border_style,
            )
        )

        if is_json and parsed_json is not None:
            json_str = json.dumps(parsed_json, indent=2)
            console.print(
                Panel(
                    Syntax(json_str, "json", theme="monokai", word_wrap=True),
                    title="📦 Response Body (Formatted JSON)",
                    border_style=border_style,
                )
            )
            # Render context-specific rich views for easier debugging
            self._render_specialized_view(parsed_json)
        else:
            display_text = raw_text[:2000] + ("\n... [truncated]" if len(raw_text) > 2000 else "")
            console.print(
                Panel(
                    display_text or "[dim](empty response body)[/dim]",
                    title="📦 Response Body (Raw Text)",
                    border_style=border_style,
                )
            )

    def _render_specialized_view(self, data: Any) -> None:
        """Render high-level readable UI cards for documents, query hits, and generated answers."""
        if not isinstance(data, dict):
            return

        # 1. Documents list
        if "documents" in data and isinstance(data["documents"], list):
            docs = data["documents"]
            table = Table(title=f"📚 Registered Documents ({len(docs)} versions)", border_style="cyan")
            table.add_column("Filename", style="bold white")
            table.add_column("Document ID", style="cyan")
            table.add_column("Version ID", style="magenta")
            table.add_column("Ver #", justify="right")
            table.add_column("Current?", justify="center")

            for doc in docs:
                is_curr = "[bold green]YES[/bold green]" if doc.get("is_current") else "[dim]no[/dim]"
                table.add_row(
                    str(doc.get("filename", "")),
                    str(doc.get("document_id", "")),
                    str(doc.get("version_id", "")),
                    str(doc.get("version_number", "")),
                    is_curr,
                )
            console.print(table)

        # 2. Query retrieval results
        results = data.get("result")
        if isinstance(results, list) and results and all(isinstance(r, dict) for r in results):
            table = Table(title=f"🎯 Retrieval Candidates ({len(results)} items)", border_style="green")
            table.add_column("Rank", justify="right", style="bold")
            table.add_column("Score", justify="right", style="green")
            table.add_column("Document ID", style="cyan")
            table.add_column("Page", justify="center", style="magenta")
            table.add_column("BBox Grounding", style="yellow")
            table.add_column("Graph Claim / Content", style="white")

            for idx, r in enumerate(results, 1):
                score_str = f"{r.get('score', 0.0):.4f}"
                if "rerank_score" in r and r["rerank_score"] is not None:
                    score_str += f" (rerank: {r['rerank_score']:.3f})"
                bbox = r.get("bbox") or r.get("highlighted_box") or "-"
                content = str(r.get("text") or r.get("claim") or r.get("chunk_text") or "-")
                if len(content) > 60:
                    content = content[:57] + "..."

                table.add_row(
                    str(idx),
                    score_str,
                    str(r.get("document_id", "-")),
                    str(r.get("page_number", "-")),
                    str(bbox),
                    content,
                )
            console.print(table)

        # 3. Multimodal Answer Generation result
        if "answer" in data and isinstance(data["answer"], str):
            answer_text = data["answer"]
            model_name = data.get("model") or "Qwen3-VL-2B-Instruct"
            latency = data.get("generation_time_ms") or data.get("latency_ms")
            cid = data.get("conversation_id")
            header_extra = f" [dim](Model: {model_name}" + (f", {latency:.1f}ms" if latency else "") + (f", Conv: {cid}" if cid else "") + ")[/dim]"

            console.print(
                Panel(
                    Markdown(answer_text),
                    title=f"💡 [bold green]Generated Multimodal Answer[/bold green]{header_extra}",
                    border_style="green",
                    padding=(1, 2),
                )
            )

        # 4. Conversations list
        if "conversations" in data and isinstance(data["conversations"], list):
            convs = data["conversations"]
            table = Table(title=f"💬 Saved Conversations ({len(convs)} items)", border_style="cyan")
            table.add_column("Conversation ID", style="bold cyan")
            table.add_column("Title", style="white")
            table.add_column("Turns", justify="center", style="yellow")
            table.add_column("Last Updated", style="dim")

            for c in convs:
                table.add_row(
                    str(c.get("conversation_id", "")),
                    str(c.get("title", "")),
                    str(c.get("turn_count", 0)),
                    str(c.get("last_message_at") or c.get("updated_at", "")),
                )
            console.print(table)

        # 5. Conversation turns history
        if "turns" in data and isinstance(data["turns"], list) and "conversation_id" in data:
            cid = data.get("conversation_id", "")
            title = data.get("title", "")
            turns = data["turns"]
            console.print(f"\n[bold cyan]💬 Conversation History:[/bold cyan] [bold white]{cid}[/bold white] — [dim]{title}[/dim] ({len(turns)} turns)\n")
            for t in turns:
                q = t.get("query", "")
                a = t.get("answer", "")
                created = t.get("created_at", "")
                console.print(
                    Panel(
                        f"[bold yellow]User:[/bold yellow] {q}\n\n[bold green]Assistant:[/bold green] {a}",
                        title=f"Turn #{t.get('turn_id', '-')} [dim]({created})[/dim]",
                        border_style="blue",
                    )
                )

        # 6. Error detail diagnosis
        if "detail" in data:
            console.print(
                Panel(
                    f"[bold red]API Error Message:[/bold red] {data['detail']}",
                    title="⚠️ Endpoint Error Detail",
                    border_style="red",
                )
            )


# ============================================================================
# Job Poller Utility
# ============================================================================

def poll_job(
    debugger: ApiDebugger,
    job_type: str,
    job_id: str,
    poll_interval: float = 2.0,
    timeout_sec: float = 180.0,
) -> Optional[dict[str, Any]]:
    """Poll an asynchronous job until completed or failed, showing a status indicator."""
    endpoint_prefix = {
        "ingest": "ingest_document",
        "query": "query",
        "generate": "generate",
    }.get(job_type, job_type)

    console.print(f"\n[cyan]⏳ Polling {job_type} job [bold]{job_id}[/bold] (timeout: {timeout_sec:.0f}s)...[/cyan]")

    start_time = time.time()
    last_status = ""

    while (time.time() - start_time) < timeout_sec:
        try:
            resp = debugger.send_request(
                "GET",
                f"{endpoint_prefix}/{job_id}",
                show_debug=False,
            )
        except Exception as exc:
            console.print(f"[yellow]Polling error: {exc}[/yellow]")
            time.sleep(poll_interval)
            continue

        if resp is None:
            time.sleep(poll_interval)
            continue

        if resp.status_code == 200:
            data = resp.json()
            status = data.get("status")

            # GET /generate/{id} returns ONLY the result object on completion
            if status is None and ("answer" in data or "result" in data):
                console.print(f"[bold green]✔ Job {job_id} COMPLETED![/bold green]")
                debugger._print_response_panel(resp, 0.0)
                return data

            if status != last_status:
                console.print(f"  [dim]{datetime.now().strftime('%H:%M:%S')}[/dim] Current Status: [bold yellow]{status}[/bold yellow]")
                last_status = status

            if status == "completed":
                console.print(f"[bold green]✔ Job {job_id} COMPLETED successfully![/bold green]")
                debugger._print_response_panel(resp, 0.0)
                return data
            elif status == "failed":
                console.print(f"[bold red]✖ Job {job_id} FAILED![/bold red]")
                debugger._print_response_panel(resp, 0.0)
                return data

        elif resp.status_code == 403:
            console.print(f"[bold red]✖ 403 Forbidden: You do not own job {job_id}.[/bold red]")
            return None
        elif resp.status_code == 404:
            console.print(f"[bold red]✖ 404 Not Found: Job {job_id} not registered yet.[/bold red]")
            return None
        else:
            console.print(f"[yellow]Unexpected status code {resp.status_code}: {resp.text}[/yellow]")

        time.sleep(poll_interval)

    console.print(f"[bold red]⏱️ Polling timed out after {timeout_sec}s for job {job_id}[/bold red]")
    return None


# ============================================================================
# Interactive CLI Endpoints Prompts
# ============================================================================

def handle_health(debugger: ApiDebugger) -> None:
    """GET /health"""
    debugger.send_request("GET", "/health")


def handle_create_user(debugger: ApiDebugger) -> None:
    """POST /create_user"""
    console.print("\n[bold cyan]👤 Mode 2: Create User Registration[/bold cyan]")
    default_name = "Test Engineer"
    default_email = f"user_{int(time.time()) % 10000}@example.com"
    default_password = "password123!"
    default_org = "Acme Corp"
    allowed_roles = ["chief", "manager", "engineer", "worker"]

    name = Prompt.ask("Full Name", default=default_name)
    email = Prompt.ask("Email", default=default_email)
    password = Prompt.ask("Password (min 8 chars)", default=default_password)
    organization = Prompt.ask("Organization", default=default_org)

    console.print(f"Available roles: [green]{', '.join(allowed_roles)}[/green]")
    role = Prompt.ask("Role", choices=allowed_roles, default="engineer")

    payload = {
        "name": name,
        "email": email,
        "password": password,
        "organization": organization,
        "role": role,
    }

    resp = debugger.send_request("POST", "/create_user", json_body=payload, use_auth=False)
    if resp is not None and resp.status_code == 200:
        data = resp.json()
        console.print(f"[bold green]✔ User created successfully![/bold green] User ID: [cyan]{data.get('user_id')}[/cyan]")
        if Confirm.ask("Would you like to login immediately with this new user?", default=True):
            _do_login(debugger, email, password)


def _do_login(debugger: ApiDebugger, email: str, password: str) -> None:
    """Helper to perform login, store JWT token, and fetch user profile."""
    payload = {"email": email, "password": password}
    resp = debugger.send_request("POST", "/login", json_body=payload, use_auth=False)
    if resp is not None and resp.status_code == 200:
        data = resp.json()
        token = data.get("access_token", "")
        user_info = data.get("user", {})
        debugger.session.set_login_data(token, user_info)
        console.print(
            Panel(
                f"[bold green]✔ Successfully Logged In & Authorized![/bold green]\n"
                f"User: [bold cyan]{user_info.get('name', 'N/A')}[/bold cyan] ({user_info.get('email')})\n"
                f"Role: [magenta]{user_info.get('role', 'N/A')}[/magenta] | Org: [yellow]{user_info.get('organization_name', 'N/A')}[/yellow]\n"
                f"User ID: [dim]{user_info.get('user_id', 'N/A')}[/dim]\n"
                f"[bold green]Saved to session file:[/bold green] [cyan]{debugger.session.session_path}[/cyan]\n"
                f"[green]All future requests will automatically include this Bearer token![/green]",
                title="🔐 Authentication Active",
                border_style="green",
            )
        )
        # Fetch /me to confirm token validity
        debugger.send_request("GET", "/me")


def handle_login(debugger: ApiDebugger) -> None:
    """POST /login"""
    console.print("\n[bold cyan]🔑 Mode 3: User Login & Token Cache[/bold cyan]")
    default_email = debugger.session.user.get("email", "testuser@example.com")
    email = Prompt.ask("Email", default=default_email)
    password = Prompt.ask("Password", default="password123!", password=False)
    _do_login(debugger, email, password)


def handle_me(debugger: ApiDebugger) -> None:
    """GET /me"""
    console.print("\n[bold cyan]👤 Mode 4: Verify Current Authenticated Identity (GET /me)[/bold cyan]")
    resp = debugger.send_request("GET", "/me")
    if resp is not None and resp.status_code == 200:
        debugger.session.user = resp.json()
        debugger.session.save()


def handle_list_documents(debugger: ApiDebugger) -> None:
    """GET /list_documents"""
    console.print("\n[bold cyan]📚 Mode 5: List Registered Documents[/bold cyan]")
    debugger.send_request("GET", "/list_documents")


def _find_available_pdfs() -> list[str]:
    """Scan local directory for PDF files to suggest to the user."""
    candidates = []
    for pattern in ["data/**/*.pdf", "*.pdf", "uploaded_pdfs/**/*.pdf"]:
        for p in BASE_DIR.glob(pattern):
            if p.is_file():
                candidates.append(str(p))
    return sorted(list(set(candidates)))


def handle_ingest_document(debugger: ApiDebugger) -> None:
    """POST /ingest_document"""
    console.print("\n[bold cyan]📥 Mode 6: Ingest Document (POST /ingest_document)[/bold cyan]")

    user_info = debugger.session.user
    default_user_id = user_info.get("user_id", "default_user")
    default_org = user_info.get("organization_name", "Acme Corp")
    default_role = user_info.get("role", "engineer")

    user_id = Prompt.ask("User ID", default=default_user_id)
    org = Prompt.ask("Organization", default=default_org)
    role = Prompt.ask("Role", default=default_role)

    source_type = Prompt.ask("Document Source", choices=["local_pdf", "s3_url", "custom_json"], default="local_pdf")

    s3_links: list[str] = []
    local_pdf_paths: list[str] = []

    if source_type == "local_pdf":
        found_pdfs = _find_available_pdfs()
        if found_pdfs:
            console.print("[dim]Detected local PDFs:[/dim]")
            for idx, p in enumerate(found_pdfs[:5], 1):
                console.print(f"  [cyan]{idx}.[/cyan] {p}")
            default_pdf = found_pdfs[0]
        else:
            default_pdf = "data/dummy/test_sample.pdf"

        pdf_path = Prompt.ask("Enter local PDF file path", default=default_pdf)
        if not Path(pdf_path).exists():
            console.print(f"[yellow]Warning: File does not exist on disk: {pdf_path}[/yellow]")
        local_pdf_paths.append(pdf_path)

    elif source_type == "s3_url":
        s3_url = Prompt.ask("Enter S3 http(s) URL", default="https://example-bucket.s3.amazonaws.com/sample.pdf")
        s3_links.append(s3_url)

    elif source_type == "custom_json":
        raw_json = Prompt.ask("Enter full JSON payload")
        try:
            payload = json.loads(raw_json)
        except Exception as exc:
            console.print(f"[red]Invalid JSON: {exc}[/red]")
            return
        resp = debugger.send_request("POST", "/ingest_document", json_body=payload)
        return

    payload = {
        "user_id": user_id,
        "organization": org,
        "role": role,
        "s3_links": s3_links,
        "local_pdf_paths": local_pdf_paths,
    }

    use_idempotency = Confirm.ask("Attach unique Idempotency-Key header?", default=True)
    custom_headers = {}
    if use_idempotency:
        custom_headers["Idempotency-Key"] = f"idem_ingest_{int(time.time())}"

    resp = debugger.send_request("POST", "/ingest_document", json_body=payload, custom_headers=custom_headers)
    if resp is not None and resp.status_code in {200, 202}:
        data = resp.json()
        job_id = data.get("job_id")
        if job_id:
            debugger.session.record_job("ingest", job_id)
            if Confirm.ask(f"Wait and automatically poll ingestion job [bold]{job_id}[/bold] until complete?", default=True):
                poll_job(debugger, "ingest", job_id)


def handle_poll_ingest(debugger: ApiDebugger) -> None:
    """GET /ingest_document/{job_id}"""
    console.print("\n[bold cyan]🔍 Mode 7: Poll Ingestion Job Status[/bold cyan]")
    default_job_id = debugger.session.last_jobs.get("ingest", "")
    job_id = Prompt.ask("Ingestion Job ID", default=default_job_id)
    if not job_id:
        console.print("[red]Job ID cannot be empty.[/red]")
        return
    debugger.session.record_job("ingest", job_id)

    if Confirm.ask("Poll continuously until complete/failed?", default=True):
        poll_job(debugger, "ingest", job_id)
    else:
        debugger.send_request("GET", f"/ingest_document/{job_id}")


def handle_query(debugger: ApiDebugger) -> None:
    """POST /query"""
    console.print("\n[bold cyan]🔎 Mode 8: Multi-Vector / Graph Query Retrieval (POST /query)[/bold cyan]")

    user_info = debugger.session.user
    user_id = Prompt.ask("User ID", default=user_info.get("user_id", "default_user"))
    org = Prompt.ask("Organization", default=user_info.get("organization_name", "Acme Corp"))
    role = Prompt.ask("Role", default=user_info.get("role", "engineer"))

    query = Prompt.ask("Search Query", default="What are the main technical conclusions and results?")
    top_k = int(Prompt.ask("Top K results", default="5"))
    recursive = Confirm.ask("Recursive search across all document versions?", default=False)
    enable_rerank = Confirm.ask("Enable multimodal Jina reranker?", default=False)
    enable_graph = Confirm.ask("Enable Knowledge Graph claims retrieval?", default=True)

    doc_ids_raw = Prompt.ask("Filter by Document IDs (comma-separated, or leave blank)", default="")
    doc_ids = [d.strip() for d in doc_ids_raw.split(",") if d.strip()]

    version_ids_raw = Prompt.ask("Filter by Version IDs (comma-separated, or leave blank)", default="")
    version_ids = [v.strip() for v in version_ids_raw.split(",") if v.strip()]

    payload: dict[str, Any] = {
        "user_id": user_id,
        "organization": org,
        "role": role,
        "query": query,
        "document_ids": doc_ids,
        "version_ids": version_ids,
        "recursive_search": recursive,
        "top_k": top_k,
        "enable_rerank": enable_rerank,
        "rerank_threshold": 0.75,
        "retrieval_top_k": 20,
        "enable_graph": enable_graph,
    }

    resp = debugger.send_request("POST", "/query", json_body=payload)
    if resp is not None and resp.status_code in {200, 202}:
        data = resp.json()
        job_id = data.get("job_id")
        if job_id:
            debugger.session.record_job("query", job_id)
            if Confirm.ask(f"Wait and poll query job [bold]{job_id}[/bold] until complete?", default=True):
                poll_job(debugger, "query", job_id)


def handle_poll_query(debugger: ApiDebugger) -> None:
    """GET /query/{query_job_id}"""
    console.print("\n[bold cyan]🔍 Mode 9: Poll Query Retrieval Job[/bold cyan]")
    default_job_id = debugger.session.last_jobs.get("query", "")
    job_id = Prompt.ask("Query Job ID", default=default_job_id)
    if not job_id:
        console.print("[red]Job ID cannot be empty.[/red]")
        return
    debugger.session.record_job("query", job_id)

    if Confirm.ask("Poll continuously until complete/failed?", default=True):
        poll_job(debugger, "query", job_id)
    else:
        debugger.send_request("GET", f"/query/{job_id}")


def handle_generate(debugger: ApiDebugger) -> None:
    """POST /generate"""
    console.print("\n[bold cyan]🧠 Mode 10: Multimodal Answer Generation (POST /generate)[/bold cyan]")

    user_info = debugger.session.user
    user_id = Prompt.ask("User ID", default=user_info.get("user_id", "default_user"))
    org = Prompt.ask("Organization", default=user_info.get("organization_name", "Acme Corp"))
    role = Prompt.ask("Role", default=user_info.get("role", "engineer"))

    query = Prompt.ask("Prompt / Question", default="Summarize the core mechanism described in the document.")
    active_cid = debugger.session.active_conversation_id
    cid_prompt = Prompt.ask(
        "Conversation ID (leave empty for new conversation)",
        default=active_cid if active_cid else "",
    ).strip()
    top_k = int(Prompt.ask("Top K context pages", default="3"))
    enable_rerank = Confirm.ask("Enable reranker for context pages?", default=False)
    enable_graph = Confirm.ask("Enable verified Knowledge Graph claims injection?", default=True)

    use_idempotency = Confirm.ask("Attach unique Idempotency-Key header?", default=True)
    custom_headers = {}
    if use_idempotency:
        custom_headers["Idempotency-Key"] = f"idem_gen_{int(time.time())}"

    payload: dict[str, Any] = {
        "user_id": user_id,
        "organization": org,
        "role": role,
        "query": query,
        "document_ids": [],
        "version_ids": [],
        "recursive_search": False,
        "top_k": top_k,
        "enable_rerank": enable_rerank,
        "rerank_threshold": 0.75,
        "retrieval_top_k": 20,
        "enable_graph": enable_graph,
    }
    if cid_prompt:
        payload["conversation_id"] = cid_prompt

    resp = debugger.send_request("POST", "/generate", json_body=payload, custom_headers=custom_headers)
    if resp is not None and resp.status_code in {200, 202}:
        data = resp.json()
        cid = data.get("conversation_id")
        if cid:
            debugger.session.active_conversation_id = cid
            debugger.session.save()
            console.print(f"[bold green]Active Conversation ID:[/bold green] [bold white]{cid}[/bold white]")
        job_id = data.get("job_id")
        if job_id:
            debugger.session.record_job("generate", job_id)
            if Confirm.ask(f"Wait and poll generation job [bold]{job_id}[/bold] until complete?", default=True):
                polled = poll_job(debugger, "generate", job_id)
                if polled and isinstance(polled, dict) and polled.get("conversation_id"):
                    debugger.session.active_conversation_id = polled["conversation_id"]
                    debugger.session.save()


def handle_poll_generate(debugger: ApiDebugger) -> None:
    """GET /generate/{generate_job_id}"""
    console.print("\n[bold cyan]🔍 Mode 11: Poll Generation Job Status[/bold cyan]")
    default_job_id = debugger.session.last_jobs.get("generate", "")
    job_id = Prompt.ask("Generate Job ID", default=default_job_id)
    if not job_id:
        console.print("[red]Job ID cannot be empty.[/red]")
        return
    debugger.session.record_job("generate", job_id)

    if Confirm.ask("Poll continuously until complete/failed?", default=True):
        poll_job(debugger, "generate", job_id)
    else:
        debugger.send_request("GET", f"/generate/{job_id}")


def handle_generate_stream(debugger: ApiDebugger) -> None:
    """POST /generate/stream (Real-Time Server-Sent Events Streaming)"""
    console.print("\n[bold cyan]⚡ Mode 12: Real-Time Streaming Generation (POST /generate/stream)[/bold cyan]")

    user_info = debugger.session.user
    user_id = Prompt.ask("User ID", default=user_info.get("user_id", "default_user"))
    org = Prompt.ask("Organization", default=user_info.get("organization_name", "Acme Corp"))
    role = Prompt.ask("Role", default=user_info.get("role", "engineer"))

    query = Prompt.ask("Prompt / Question", default="Summarize the core technical findings in the document.")
    active_cid = debugger.session.active_conversation_id
    cid_prompt = Prompt.ask(
        "Conversation ID (leave empty for new conversation)",
        default=active_cid if active_cid else "",
    ).strip()
    top_k = int(Prompt.ask("Top K context pages", default="3"))
    enable_rerank = Confirm.ask("Enable reranker for context pages?", default=False)
    enable_graph = Confirm.ask("Enable verified Knowledge Graph claims injection?", default=True)

    payload: dict[str, Any] = {
        "user_id": user_id,
        "organization": org,
        "role": role,
        "query": query,
        "document_ids": [],
        "version_ids": [],
        "recursive_search": False,
        "top_k": top_k,
        "enable_rerank": enable_rerank,
        "rerank_threshold": 0.75,
        "retrieval_top_k": 20,
        "enable_graph": enable_graph,
    }
    if cid_prompt:
        payload["conversation_id"] = cid_prompt

    url = f"{debugger.session.base_url}/generate/stream"
    headers = {
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
        "User-Agent": "AMG-CLI-Debugger/1.0",
    }
    headers.update(debugger.session.get_auth_headers())

    debugger._print_request_panel("POST", url, headers, None, payload)

    console.print("\n[cyan]Connecting to SSE stream and generating tokens...[/cyan]\n")

    start_time = time.perf_counter()
    first_token_time: Optional[float] = None
    accumulated_tokens: list[str] = []
    current_event: Optional[str] = None
    complete_payload: Optional[dict[str, Any]] = None

    try:
        with requests.post(url, json=payload, headers=headers, stream=True, timeout=120) as resp:
            if resp.status_code != 200:
                console.print(f"[bold red]✖ Stream request failed with status {resp.status_code}[/bold red]")
                try:
                    err_json = resp.json()
                    debugger._render_specialized_view(err_json)
                except Exception:
                    console.print(f"[red]{resp.text}[/red]")
                return

            for line_bytes in resp.iter_lines(decode_unicode=True):
                if not line_bytes:
                    continue
                line = line_bytes.strip()
                if line.startswith("event:"):
                    current_event = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    raw_data = line[len("data:"):].strip()
                    try:
                        data_obj = json.loads(raw_data)
                    except Exception:
                        data_obj = {"raw": raw_data}

                    if current_event == "token":
                        token_str = data_obj.get("token", "")
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        accumulated_tokens.append(token_str)
                        sys.stdout.write(token_str)
                        sys.stdout.flush()

                    elif current_event == "complete":
                        complete_payload = data_obj
                    elif current_event == "error":
                        console.print(f"\n[bold red]✖ Stream Error Event:[/bold red] {data_obj}")

        sys.stdout.write("\n\n")
        total_time = (time.perf_counter() - start_time) * 1000.0
        ttft_ms = ((first_token_time - start_time) * 1000.0) if first_token_time else 0.0

        if complete_payload:
            meta = complete_payload.get("generation", {})
            dur = meta.get("duration_ms") or total_time
            ttft = complete_payload.get("ttft_ms") or ttft_ms

            summary_text = (
                f"[bold green]✔ Stream Complete![/bold green]  "
                f"[bold cyan]Total Time:[/bold cyan] {dur:.1f}ms  "
                f"[bold cyan]TTFT:[/bold cyan] {ttft:.1f}ms  "
                f"[bold cyan]Tokens Streamed:[/bold cyan] {len(accumulated_tokens)}"
            )
            console.print(Panel(summary_text, title="⚡ Streaming Performance", border_style="green"))

            # Display sources if present
            sources = complete_payload.get("sources")
            if sources and isinstance(sources, list):
                s_table = Table(title="📖 Grounded Evidence Sources", border_style="cyan")
                s_table.add_column("Source", style="bold yellow")
                s_table.add_column("Rank", justify="center")
                s_table.add_column("Score", justify="right", style="green")
                s_table.add_column("Document ID", style="cyan")
                s_table.add_column("Page", justify="center", style="magenta")
                for s in sources:
                    s_table.add_row(
                        str(s.get("source_id", "-")),
                        str(s.get("rank", "-")),
                        f"{s.get('score', 0.0):.4f}",
                        str(s.get("document_id", "-")),
                        str(s.get("page_number", "-")),
                    )
                console.print(s_table)

            cid = complete_payload.get("conversation_id")
            if cid:
                debugger.session.active_conversation_id = cid
                debugger.session.save()
                console.print(f"\n[bold green]Active Conversation ID:[/bold green] [bold white]{cid}[/bold white]")
        else:
            console.print(f"[green]Stream ended. Total latency: {total_time:.1f}ms[/green]")

    except Exception as exc:
        console.print(f"\n[bold red]Streaming error:[/bold red] {exc}")


def handle_conversations(debugger: ApiDebugger) -> None:
    """Mode 13: Conversation History & Context Management"""
    console.print("\n[bold cyan]💬 Mode 13: Conversation History & Context Management[/bold cyan]")
    console.print("  [cyan]1[/cyan]. List all conversations (GET /conversations)")
    console.print("  [cyan]2[/cyan]. View conversation history & turns (GET /conversations/{id})")
    console.print("  [cyan]3[/cyan]. Delete conversation (DELETE /conversations/{id})")
    console.print("  [cyan]4[/cyan]. Switch / Clear active conversation ID")

    action = Prompt.ask("Choose action", choices=["1", "2", "3", "4"], default="1")

    if action == "1":
        limit = int(Prompt.ask("Max conversations to list", default="20"))
        debugger.send_request("GET", f"/conversations?limit={limit}")

    elif action == "2":
        default_id = debugger.session.active_conversation_id
        cid = Prompt.ask("Conversation ID", default=default_id).strip()
        if not cid:
            console.print("[red]Conversation ID cannot be empty.[/red]")
            return
        resp = debugger.send_request("GET", f"/conversations/{cid}")
        if resp is not None and resp.status_code == 200:
            if Confirm.ask("Set this conversation as the active conversation?", default=True):
                debugger.session.active_conversation_id = cid
                debugger.session.save()
                console.print(f"[green]Active conversation set to:[/green] [bold]{cid}[/bold]")

    elif action == "3":
        default_id = debugger.session.active_conversation_id
        cid = Prompt.ask("Conversation ID to DELETE", default=default_id).strip()
        if not cid:
            console.print("[red]Conversation ID cannot be empty.[/red]")
            return
        if Confirm.ask(f"[bold red]Are you sure you want to delete conversation {cid} and ALL its turns?[/bold red]", default=False):
            resp = debugger.send_request("DELETE", f"/conversations/{cid}")
            if resp is not None and resp.status_code == 200:
                if debugger.session.active_conversation_id == cid:
                    debugger.session.active_conversation_id = ""
                    debugger.session.save()

    elif action == "4":
        current = debugger.session.active_conversation_id
        console.print(f"Current active conversation ID: [bold]{current or '(none)'}[/bold]")
        new_cid = Prompt.ask("Enter new active conversation ID (or 'clear' to unset)", default=current).strip()
        if new_cid.lower() in {"clear", "unset", "none", ""}:
            debugger.session.active_conversation_id = ""
            console.print("[yellow]Active conversation cleared.[/yellow]")
        else:
            debugger.session.active_conversation_id = new_cid
            console.print(f"[green]Active conversation set to:[/green] [bold]{new_cid}[/bold]")
        debugger.session.save()


def handle_smoke_test(debugger: ApiDebugger) -> None:
    """Mode 12: Automated Smoke Test across core endpoints."""
    console.print(
        Panel(
            "[bold cyan]🧪 Starting End-to-End API Smoke Test Suite[/bold cyan]\n"
            "Will sequentially test: Health -> User Registration -> Login & Token -> Auth Check (/me) -> List Documents",
            title="Smoke Test",
            border_style="magenta",
        )
    )

    # 1. Health
    console.print("\n[bold cyan]Step 1/5: Checking API Health...[/bold cyan]")
    h_resp = debugger.send_request("GET", "/health")
    if h_resp is None or h_resp.status_code != 200:
        console.print("[bold red]✖ Smoke test failed at Step 1: Health check failed.[/bold red]")
        return

    # 2. Register new test user
    rand_id = int(time.time()) % 100000
    test_email = f"smoketest_{rand_id}@example.com"
    test_pass = "SmokePass123!"
    console.print(f"\n[bold cyan]Step 2/5: Registering new test user ({test_email})...[/bold cyan]")
    reg_payload = {
        "name": f"Smoke Tester {rand_id}",
        "email": test_email,
        "password": test_pass,
        "organization": "Smoke Test Org",
        "role": "engineer",
    }
    r_resp = debugger.send_request("POST", "/create_user", json_body=reg_payload, use_auth=False)
    if r_resp is None or r_resp.status_code != 200:
        console.print("[bold red]✖ Smoke test failed at Step 2: User registration failed.[/bold red]")
        return

    # 3. Login
    console.print("\n[bold cyan]Step 3/5: Logging in to issue JWT token...[/bold cyan]")
    _do_login(debugger, test_email, test_pass)
    if not debugger.session.is_authenticated():
        console.print("[bold red]✖ Smoke test failed at Step 3: Login failed to acquire JWT token.[/bold red]")
        return

    # 4. Identity
    console.print("\n[bold cyan]Step 4/5: Verifying JWT identity via /me...[/bold cyan]")
    me_resp = debugger.send_request("GET", "/me")
    if me_resp is None or me_resp.status_code != 200:
        console.print("[bold red]✖ Smoke test failed at Step 4: /me returned non-200.[/bold red]")
        return

    # 5. List docs
    console.print("\n[bold cyan]Step 5/5: Listing registered documents...[/bold cyan]")
    doc_resp = debugger.send_request("GET", "/list_documents")
    if doc_resp is None or doc_resp.status_code != 200:
        console.print("[bold red]✖ Smoke test failed at Step 5: /list_documents failed.[/bold red]")
        return

    console.print(
        Panel(
            "[bold green]✔ All Smoke Tests Passed Successfully![/bold green]\n"
            "The API is live, Redis is reachable, and the JWT token was saved and verified.",
            title="Smoke Test Result",
            border_style="green",
        )
    )


def handle_session_settings(debugger: ApiDebugger) -> None:
    """Mode 13: Session / Auth Management"""
    console.print("\n[bold cyan]⚙️ Mode 13: Session & Token Management[/bold cyan]")

    s = debugger.session
    claims = s.decode_token_unverified()

    info_table = Table(title="Current Session Profile", border_style="cyan")
    info_table.add_column("Property", style="bold yellow")
    info_table.add_column("Value", style="white")

    info_table.add_row("API Base URL", s.base_url)
    info_table.add_row("Session File", str(s.session_path))
    info_table.add_row("Auto-Attach JWT?", "[green]Enabled[/green]" if s.auto_auth else "[red]Disabled[/red]")
    info_table.add_row("Authenticated?", "[bold green]YES[/bold green]" if s.is_authenticated() else "[dim red]NO[/dim red]")

    if s.user:
        info_table.add_row("User ID", str(s.user.get("user_id", "N/A")))
        info_table.add_row("Email", str(s.user.get("email", "N/A")))
        info_table.add_row("Name", str(s.user.get("name", "N/A")))
        info_table.add_row("Role", str(s.user.get("role", "N/A")))
        info_table.add_row("Organization", str(s.user.get("organization_name", "N/A")))

    if claims:
        info_table.add_row("JWT Subject (sub)", str(claims.get("sub", "N/A")))
        info_table.add_row("JWT Issuer (iss)", str(claims.get("iss", "N/A")))
        info_table.add_row("JWT Audience (aud)", str(claims.get("aud", "N/A")))
        if "exp" in claims:
            exp_dt = datetime.fromtimestamp(claims["exp"], tz=timezone.utc).isoformat()
            info_table.add_row("JWT Expiration (exp)", f"{exp_dt} ({claims['exp']})")

    console.print(info_table)

    console.print("\nAvailable Actions:")
    console.print("  [cyan]1[/cyan]. Change Base URL")
    console.print("  [cyan]2[/cyan]. Paste / Manually Set Access Token")
    console.print("  [cyan]3[/cyan]. Toggle Auto-Attach Token (test 401 Unauthorized scenarios)")
    console.print("  [cyan]4[/cyan]. Clear Session & Logout")
    console.print("  [cyan]0[/cyan]. Back to Main Menu")

    action = Prompt.ask("Choose action", choices=["1", "2", "3", "4", "0"], default="0")
    if action == "1":
        new_url = Prompt.ask("Enter new API Base URL", default=s.base_url).rstrip("/")
        s.base_url = new_url
        s.save()
        console.print(f"[green]API Base URL updated to: {s.base_url}[/green]")
    elif action == "2":
        new_token = Prompt.ask("Paste Bearer JWT token")
        s.set_login_data(new_token)
        console.print("[green]Token updated. Checking /me...[/green]")
        debugger.send_request("GET", "/me")
    elif action == "3":
        s.auto_auth = not s.auto_auth
        s.save()
        status_msg = "ENABLED" if s.auto_auth else "DISABLED"
        console.print(f"[yellow]Auto-attach JWT is now {status_msg}.[/yellow]")
    elif action == "4":
        s.clear()
        console.print("[yellow]Session cleared. You are now unauthenticated.[/yellow]")


def handle_custom_request(debugger: ApiDebugger) -> None:
    """Mode 14: Custom Raw Request"""
    console.print("\n[bold cyan]🔧 Mode 14: Custom Raw Request[/bold cyan]")

    method = Prompt.ask("HTTP Method", choices=["GET", "POST", "PUT", "DELETE", "PATCH"], default="GET")
    path = Prompt.ask("Endpoint Path (e.g. /health, /query/my_id)", default="/health")

    include_auth = Confirm.ask("Include Bearer Authorization token?", default=debugger.session.auto_auth)

    json_body = None
    if method in {"POST", "PUT", "PATCH"}:
        body_input = Prompt.ask("JSON Body (leave empty for none)", default="")
        if body_input.strip():
            try:
                json_body = json.loads(body_input)
            except Exception as exc:
                console.print(f"[red]Error parsing JSON: {exc}[/red]")
                return

    debugger.send_request(method, path, json_body=json_body, use_auth=include_auth)


# ============================================================================
# Main Interactive Menu & Loop
# ============================================================================

def print_banner(session: SessionManager) -> None:
    """Display an overview banner showing server connectivity and authenticated user."""
    auth_status = (
        f"[bold green]✔ Authorized[/bold green] as [bold cyan]{session.user.get('name', 'User')}[/bold cyan] "
        f"([magenta]{session.user.get('role', 'N/A')}[/magenta] @ [yellow]{session.user.get('organization_name', 'N/A')}[/yellow])"
        if session.is_authenticated()
        else "[dim yellow]⚠️ Unauthenticated (No JWT token loaded - Login via Mode 3)[/dim yellow]"
    )
    active_conv = (
        f"[bold cyan]{session.active_conversation_id}[/bold cyan]"
        if session.active_conversation_id
        else "[dim](none - new conversation)[/dim]"
    )

    console.print(
        Panel(
            f"[bold white]Target Server:[/bold white] [cyan]{session.base_url}[/cyan]\n"
            f"[bold white]Auth Status:[/bold white]   {auth_status}\n"
            f"[bold white]Active Conv:[/bold white]   {active_conv}\n"
            f"[bold white]Session File:[/bold white]  [dim]{session.session_path}[/dim]",
            title="[bold blue]AMG Multimodal RAG — API Debugger & Test CLI[/bold blue]",
            border_style="cyan",
            padding=(0, 2),
        )
    )


def interactive_loop(session: SessionManager) -> None:
    """Main interactive menu loop."""
    debugger = ApiDebugger(session)

    while True:
        console.print()
        print_banner(session)

        console.print("\n[bold white]Select an Endpoint Mode:[/bold white]")
        console.print("  [cyan]1[/cyan].  GET  /health                   [dim]Server & Redis Health Check[/dim]")
        console.print("  [cyan]2[/cyan].  POST /create_user              [dim]Register New User / Organization[/dim]")
        console.print("  [cyan]3[/cyan].  POST /login                    [dim]Authenticate & Auto-Save JWT Token[/dim]")
        console.print("  [cyan]4[/cyan].  GET  /me                       [dim]Current Authenticated User Profile[/dim]")
        console.print("  [cyan]5[/cyan].  GET  /list_documents           [dim]Registered Documents & Versions[/dim]")
        console.print("  [cyan]6[/cyan].  POST /ingest_document          [dim]Submit Document Ingestion Job[/dim]")
        console.print("  [cyan]7[/cyan].  GET  /ingest_document/{job_id} [dim]Poll Document Ingestion Status[/dim]")
        console.print("  [cyan]8[/cyan].  POST /query                    [dim]Multi-Vector / Graph Retrieval[/dim]")
        console.print("  [cyan]9[/cyan].  GET  /query/{query_job_id}     [dim]Poll Query Job Retrieval Results[/dim]")
        console.print("  [cyan]10[/cyan]. POST /generate                 [dim]Multimodal Answer Generation (Async Job)[/dim]")
        console.print("  [cyan]11[/cyan]. GET  /generate/{job_id}         [dim]Poll Generation Job Results[/dim]")
        console.print("  [green]12[/green]. POST /generate/stream          [dim]Real-Time SSE Streaming Generation[/dim]")
        console.print("  [cyan]13[/cyan]. GET  /conversations            [dim]Conversation History & Turns[/dim]")
        console.print("  [magenta]14[/magenta]. Automated Smoke Test Suite       [dim]Verify non-destructive endpoints in order[/dim]")
        console.print("  [yellow]15[/yellow]. Session / Auth Management        [dim]Inspect token, switch user, change URL[/dim]")
        console.print("  [blue]16[/blue]. Custom Raw HTTP Request        [dim]Arbitrary method, endpoint, and payload[/dim]")
        console.print("  [red]0[/red].  Exit")

        choice = Prompt.ask("\nEnter Mode #", default="1")
        console.print()

        try:
            if choice == "1":
                handle_health(debugger)
            elif choice == "2":
                handle_create_user(debugger)
            elif choice == "3":
                handle_login(debugger)
            elif choice == "4":
                handle_me(debugger)
            elif choice == "5":
                handle_list_documents(debugger)
            elif choice == "6":
                handle_ingest_document(debugger)
            elif choice == "7":
                handle_poll_ingest(debugger)
            elif choice == "8":
                handle_query(debugger)
            elif choice == "9":
                handle_poll_query(debugger)
            elif choice == "10":
                handle_generate(debugger)
            elif choice == "11":
                handle_poll_generate(debugger)
            elif choice == "12":
                handle_generate_stream(debugger)
            elif choice == "13":
                handle_conversations(debugger)
            elif choice == "14":
                handle_smoke_test(debugger)
            elif choice == "15":
                handle_session_settings(debugger)
            elif choice == "16":
                handle_custom_request(debugger)
            elif choice in {"0", "q", "exit", "quit"}:
                console.print("[cyan]Goodbye![/cyan]")
                break
            else:
                console.print(f"[red]Unknown mode: {choice}[/red]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        except Exception as exc:
            console.print(f"\n[bold red]Error executing mode {choice}:[/bold red] {exc}")


# ============================================================================
# CLI Flag Parser (Non-Interactive Automation)
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AMG Multimodal RAG — API Interactive Debugger & Test CLI"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="API Base URL (e.g. http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Explicit JWT token to save and use",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=[
            "health", "create_user", "login", "me", "list_documents",
            "ingest", "query", "generate", "generate_stream", "smoke_test",
        ],
        help="Execute a specific mode directly without interactive menu",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Query text when invoking query or generate mode",
    )
    parser.add_argument(
        "--poll",
        action="store_true",
        help="Automatically poll jobs to completion when submitting ingest/query/generate",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run full automated smoke test suite immediately",
    )

    args = parser.parse_args()

    session = SessionManager()

    if args.url:
        session.base_url = args.url.rstrip("/")
        session.save()

    if args.token:
        session.set_login_data(args.token)

    debugger = ApiDebugger(session)

    # 1. Smoke test flag
    if args.smoke_test:
        handle_smoke_test(debugger)
        return

    # 2. Non-interactive direct mode
    if args.mode:
        if args.mode == "health":
            handle_health(debugger)
        elif args.mode == "me":
            handle_me(debugger)
        elif args.mode == "list_documents":
            handle_list_documents(debugger)
        elif args.mode == "smoke_test":
            handle_smoke_test(debugger)
        elif args.mode == "query":
            q_text = args.query or "What are the key technical findings in the document?"
            payload = {
                "user_id": session.user.get("user_id", "default_user"),
                "organization": session.user.get("organization_name", "Acme Corp"),
                "role": session.user.get("role", "engineer"),
                "query": q_text,
                "document_ids": [],
                "version_ids": [],
                "recursive_search": False,
                "top_k": 5,
                "enable_rerank": False,
                "rerank_threshold": 0.75,
                "retrieval_top_k": 20,
                "enable_graph": True,
            }
            resp = debugger.send_request("POST", "/query", json_body=payload)
            if args.poll and resp is not None and resp.status_code in {200, 202}:
                poll_job(debugger, "query", resp.json().get("job_id"))
        elif args.mode == "generate":
            q_text = args.query or "Summarize the technical contents of the document."
            payload = {
                "user_id": session.user.get("user_id", "default_user"),
                "organization": session.user.get("organization_name", "Acme Corp"),
                "role": session.user.get("role", "engineer"),
                "query": q_text,
                "document_ids": [],
                "version_ids": [],
                "recursive_search": False,
                "top_k": 3,
                "enable_rerank": False,
                "rerank_threshold": 0.75,
                "retrieval_top_k": 20,
                "enable_graph": True,
            }
            resp = debugger.send_request("POST", "/generate", json_body=payload)
            if args.poll and resp is not None and resp.status_code in {200, 202}:
                poll_job(debugger, "generate", resp.json().get("job_id"))
        elif args.mode == "generate_stream":
            handle_generate_stream(debugger)
        return

    # 3. Default to interactive menu
    interactive_loop(session)


if __name__ == "__main__":
    main()
