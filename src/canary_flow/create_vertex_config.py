"""Shared LLM factory for Vertex AI / Gemini models."""

import base64
import json
import os
from pathlib import Path

from crewai import LLM
from google.genai import types

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

# Prevent silent safety-filter blocking — let all categories through.
_SAFETY_SETTINGS = [
    types.SafetySetting(category=cat, threshold="OFF")
    for cat in (
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    )
]


def _get_vertex_credentials():
    """Load Vertex AI credentials + project from service account.

    Returns (credentials, project_id) tuple. project_id is extracted
    from the service account JSON when GOOGLE_CLOUD_PROJECT is not set.
    """
    from google.oauth2 import service_account

    sa_base64 = os.getenv("GOOGLE_SERVICE_ACCOUNT_BASE64")
    if sa_base64:
        sa_info = json.loads(base64.b64decode(sa_base64))
        creds = service_account.Credentials.from_service_account_info(
            sa_info, scopes=_SCOPES
        )
        project = os.getenv("GOOGLE_CLOUD_PROJECT") or sa_info.get("project_id")
        return creds, project

    return None, os.getenv("GOOGLE_CLOUD_PROJECT")


def create_llm(model: str = "gemini-3.1-flash-lite-preview") -> LLM:
    """Create a Gemini LLM instance.

    Prefers Vertex AI with service account credentials (always works).
    Falls back to API key auth if no service account is available.

    GeminiCompletion auto-reads GOOGLE_API_KEY from env, which conflicts
    with service account credentials, so we temporarily hide it.
    """
    credentials, project = _get_vertex_credentials()

    if credentials:
        return LLM(
            model=model,
            provider="gemini",
            project=project,
            location="global",
            temperature=0.3,
            safety_settings=_SAFETY_SETTINGS,
            client_params={"credentials": credentials},
        )
  