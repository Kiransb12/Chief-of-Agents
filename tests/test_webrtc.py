"""
Tests for WebRTC Signaling and Dual DataChannel Voice Bridge.
"""
import pytest
import asyncio
from fastapi.testclient import TestClient
from aiortc import RTCPeerConnection, RTCSessionDescription
from app.main import app


@pytest.mark.asyncio
async def test_webrtc_offer_signaling():
    """Verifies that POST /webrtc/offer accepts an SDP offer and returns a valid SDP answer."""
    pc = RTCPeerConnection()
    pc.createDataChannel("media_channel")
    pc.createDataChannel("live_updates")

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    client = TestClient(app)
    response = client.post(
        "/webrtc/offer",
        json={
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "session_id": "test-webrtc-session-123"
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert "sdp" in data
    assert data["type"] == "answer"
    assert data["session_id"] == "test-webrtc-session-123"

    await pc.close()
