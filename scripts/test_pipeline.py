#!/usr/bin/env python3
"""
Standalone test script for Swish-Vision pipeline.
Tests the complete flow: login -> upload -> poll for completion.
"""

import argparse
import os
import requests
import time
import sys

# Configuration - override via environment variables or CLI args
BASE_URL = os.getenv("SWISHVISION_BASE_URL", "http://localhost:8000")
TEST_EMAIL = os.getenv("SWISHVISION_TEST_EMAIL", "test@example.com")
TEST_PASSWORD = os.getenv("SWISHVISION_TEST_PASSWORD")
TEST_VIDEO_PATH = os.getenv("SWISHVISION_TEST_VIDEO")

# Constants
REGISTER_ENDPOINT = f"{BASE_URL}/api/auth/register"
LOGIN_ENDPOINT = f"{BASE_URL}/api/auth/login"
UPLOAD_ENDPOINT = f"{BASE_URL}/api/sessions/upload"
SESSION_ENDPOINT = f"{BASE_URL}/api/sessions"
POLL_INTERVAL = 5   # seconds
POLL_TIMEOUT = 600  # 10 minutes max
REQUEST_TIMEOUT = 30   # seconds for regular requests
UPLOAD_TIMEOUT = 300   # seconds for file upload


def register():
    """
    Register test user (or skip if already exists).
    
    Returns:
        bool: True if registration succeeded or user already exists, False otherwise
    """
    print(f"\n[0] Registering test user: {TEST_EMAIL}")
    
    payload = {
        "name": "Test User",
        "email": TEST_EMAIL,
        "password": TEST_PASSWORD
    }
    
    try:
        response = requests.post(REGISTER_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT)
        
        if response.status_code == 409:
            print(f"✓ User already exists")
            return True
        
        response.raise_for_status()
        print(f"✓ Registration successful")
        return True
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Registration request failed: {e}")
        return False


def login():
    """
    Log in and return JWT token.
    
    Returns:
        str: JWT token if successful, None otherwise
    """
    print(f"\n[1] Logging in with credentials: {TEST_EMAIL}")
    
    payload = {
        "email": TEST_EMAIL,
        "password": TEST_PASSWORD
    }
    
    try:
        response = requests.post(LOGIN_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Login request failed: {e}")
        return None
    
    try:
        data = response.json()
        token = data.get("access_token")
        if not token:
            print(f"ERROR: No access_token in response: {data}")
            return None
        print(f"✓ Login successful")
        return token
    except ValueError as e:
        print(f"ERROR: Failed to parse login response: {e}")
        return None


def upload_video(token):
    """
    Upload test video file and return session ID and status.
    
    Args:
        token (str): JWT Bearer token
        
    Returns:
        tuple: (session_id, status) if successful, (None, None) otherwise
    """
    print(f"\n[2] Uploading video: {TEST_VIDEO_PATH}")
    
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    try:
        with open(TEST_VIDEO_PATH, "rb") as f:
            files = {"file": (TEST_VIDEO_PATH.split("\\")[-1], f, "video/mp4")}
            response = requests.post(UPLOAD_ENDPOINT, files=files, headers=headers, timeout=UPLOAD_TIMEOUT)
            response.raise_for_status()
    except FileNotFoundError:
        print(f"ERROR: Test video file not found: {TEST_VIDEO_PATH}")
        return None, None
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Upload request failed: {e}")
        try:
            error_detail = response.json()
            print(f"  Server response: {error_detail}")
        except:
            print(f"  Response text: {response.text}")
        return None, None
    
    try:
        data = response.json()
        session_id = data.get("id")
        status = data.get("status")
        
        if not session_id:
            print(f"ERROR: No session_id in response: {data}")
            return None, None
        
        print(f"✓ Upload successful")
        print(f"  Session ID: {session_id}")
        print(f"  Initial Status: {status}")
        return session_id, status
    except ValueError as e:
        print(f"ERROR: Failed to parse upload response: {e}")
        return None, None


def poll_session_status(session_id, token):
    """
    Poll session status until completion or failure.
    
    Args:
        session_id (str): Session ID to poll
        token (str): JWT Bearer token
        
    Returns:
        dict: Final session details if completed/failed, None if timeout
    """
    print(f"\n[3] Polling session status (interval: {POLL_INTERVAL}s, timeout: {POLL_TIMEOUT}s)")
    
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    start_time = time.time()
    poll_count = 0
    
    while True:
        elapsed = time.time() - start_time
        
        if elapsed > POLL_TIMEOUT:
            print(f"\nERROR: Polling timeout ({POLL_TIMEOUT}s) exceeded")
            return None
        
        try:
            response = requests.get(
                f"{SESSION_ENDPOINT}/{session_id}",
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Polling request failed: {e}")
            return None
        
        try:
            data = response.json()
            status = data.get("status")
            poll_count += 1
            
            print(f"  Poll #{poll_count} ({elapsed:.1f}s): {status}")
            
            if status in ["completed", "failed"]:
                return data
            
            time.sleep(POLL_INTERVAL)
        except ValueError as e:
            print(f"ERROR: Failed to parse poll response: {e}")
            return None


def main():
    """Run the complete test pipeline."""
    parser = argparse.ArgumentParser(description="Test the Swish-Vision pipeline end-to-end.")
    parser.add_argument(
        "--base-url",
        default=None,
        help="API base URL (overrides SWISHVISION_BASE_URL; default: http://localhost:8000)",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Path to test video file (overrides SWISHVISION_TEST_VIDEO env var)",
    )
    parser.add_argument(
        "--email",
        default=None,
        help="Test account email (overrides SWISHVISION_TEST_EMAIL; default: test@example.com)",
    )
    parser.add_argument(
        "--password",
        default=None,
        help="Test account password (overrides SWISHVISION_TEST_PASSWORD env var)",
    )
    args = parser.parse_args()

    global BASE_URL, TEST_EMAIL, TEST_PASSWORD, TEST_VIDEO_PATH, REGISTER_ENDPOINT, LOGIN_ENDPOINT, UPLOAD_ENDPOINT, SESSION_ENDPOINT
    if args.base_url:
        BASE_URL = args.base_url
        REGISTER_ENDPOINT = f"{BASE_URL}/api/auth/register"
        LOGIN_ENDPOINT = f"{BASE_URL}/api/auth/login"
        UPLOAD_ENDPOINT = f"{BASE_URL}/api/sessions/upload"
        SESSION_ENDPOINT = f"{BASE_URL}/api/sessions"
    if args.email:
        TEST_EMAIL = args.email
    if args.password:
        TEST_PASSWORD = args.password
    if args.video:
        TEST_VIDEO_PATH = args.video

    if not TEST_PASSWORD:
        print("ERROR: No test password provided. Use --password or set SWISHVISION_TEST_PASSWORD.")
        return 1

    if not TEST_VIDEO_PATH:
        print("ERROR: No test video path provided. Use --video or set SWISHVISION_TEST_VIDEO.")
        return 1

    print("=" * 60)
    print("Swish-Vision Pipeline Test")
    print("=" * 60)
    
    # Step 0: Register (if needed)
    if not register():
        print("\n✗ Test failed: Could not register user")
        return 1
    
    # Step 1: Login
    token = login()
    if not token:
        print("\n✗ Test failed: Could not obtain token")
        return 1
    
    # Step 2: Upload video
    session_id, initial_status = upload_video(token)
    if not session_id:
        print("\n✗ Test failed: Could not upload video")
        return 1
    
    # Step 3: Poll for completion
    final_session = poll_session_status(session_id, token)
    if not final_session:
        print("\n✗ Test failed: Polling timeout or error")
        return 1
    
    # Summary
    print(f"\n[4] Final Session Details")
    print(f"=" * 60)
    for key, value in final_session.items():
        if key not in ["video_data", "frames"]:  # Skip large nested data
            print(f"  {key}: {value}")
    
    final_status = final_session.get("status")
    if final_status == "completed":
        print("\n✓ Test completed successfully!")
        return 0
    else:
        print(f"\n✗ Test ended with status: {final_status}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
