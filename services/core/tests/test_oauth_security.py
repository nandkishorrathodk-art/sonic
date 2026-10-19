"""OAuth trust boundaries: identity verification and same-origin redirects."""
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlsplit

import pytest

from sonic.api.routes import auth
from sonic.auth import google_auth
from sonic.auth.models import AuthToken, User, UserRole


@pytest.mark.asyncio
@pytest.mark.parametrize('verified', [False, None, 'true', 1])
async def test_google_rejects_unverified_identity(monkeypatch, verified):
    monkeypatch.setattr(google_auth, 'exchange_code_for_tokens', AsyncMock(return_value={'access_token': 'local'}))
    monkeypatch.setattr(google_auth, 'fetch_google_user_info', AsyncMock(return_value={
        'email': 'admin@example.com', 'verified_email': verified,
    }))
    monkeypatch.setattr(google_auth, 'is_email_allowed', lambda email: True)
    with pytest.raises(PermissionError, match='verified'):
        await google_auth.authenticate_with_google('code')


@pytest.mark.asyncio
async def test_google_accepts_verified_allowlisted_identity(monkeypatch):
    monkeypatch.setattr(google_auth, 'exchange_code_for_tokens', AsyncMock(return_value={'access_token': 'local'}))
    monkeypatch.setattr(google_auth, 'fetch_google_user_info', AsyncMock(return_value={
        'email': 'operator@example.com', 'name': 'Operator', 'verified_email': True,
    }))
    monkeypatch.setattr(google_auth, 'is_email_allowed', lambda email: True)
    result = await google_auth.authenticate_with_google('code')
    assert result.user.email == 'operator@example.com'


@pytest.mark.asyncio
@pytest.mark.parametrize('redirect', ['/\\evil.test', '//evil.test', '/\t/evil.test', 'https://evil.test'])
async def test_callback_rejects_external_redirect_forms(monkeypatch, redirect):
    monkeypatch.setattr(auth, 'consume_oauth_state', lambda state: redirect)
    monkeypatch.setattr(auth, 'authenticate_with_google', AsyncMock(return_value=AuthToken(
        access_token='local-token', expires_in=3600, user=User(email='u@example.com', name='U', role=UserRole.OPERATOR),
    )))
    result = await auth.google_callback(code='code', state='local-state')
    assert urlsplit(result.headers['location']).path == '/dashboard'


@pytest.mark.asyncio
async def test_callback_encodes_query_before_fragment(monkeypatch):
    monkeypatch.setattr(auth, 'consume_oauth_state', lambda state: '/dashboard?view=tasks&token=stale&role=super_admin#active')
    monkeypatch.setattr(auth, 'authenticate_with_google', AsyncMock(return_value=AuthToken(
        access_token='local-token', expires_in=3600, user=User(email='u@example.com', name='A & B', role=UserRole.OPERATOR, tenant_id='team&role=super_admin'),
    )))
    result = await auth.google_callback(code='code', state='local-state')
    target = urlsplit(result.headers['location'])
    query = parse_qs(target.query)
    assert target.fragment == 'active'
    assert query['token'] == ['local-token']
    assert query['view'] == ['tasks']
    assert query['name'] == ['A & B']
    assert query['tenant_id'] == ['team&role=super_admin']
    assert query['role'] == ['operator']
