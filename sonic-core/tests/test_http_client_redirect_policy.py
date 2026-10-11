import pytest

from sonic.tools.adapters.http_adapter import HTTPClientAdapter
from sonic.tools.base import ToolRequest


def _request(options):
    return ToolRequest(
        tenant_id="tenant",
        engagement_id="engagement",
        workspace_id="workspace",
        agent_id="agent",
        tool_name="http_client",
        target="https://target.example.test",
        options=options,
    )


def test_http_adapter_does_not_follow_redirects_by_default():
    command = HTTPClientAdapter(provider=None).build_command(_request({}))

    assert "-L" not in command


def test_http_adapter_rejects_unvalidated_redirect_following():
    adapter = HTTPClientAdapter(provider=None)

    with pytest.raises(ValueError, match="scope- and egress-validated"):
        adapter.build_command(_request({"follow_redirects": True}))


def test_http_adapter_rejects_non_boolean_redirect_option():
    adapter = HTTPClientAdapter(provider=None)

    with pytest.raises(ValueError, match="must be a boolean"):
        adapter.build_command(_request({"follow_redirects": "yes"}))
