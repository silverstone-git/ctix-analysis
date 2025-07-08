import webbrowser
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import asyncio
import threading
import os
import logging
from kiteconnect import KiteConnect

# --- Configuration ---
PORT = 8000

# --- FastAPI App ---
app = FastAPI()
request_token_received = None # Global variable to store the request_token
server_instance = None # To hold the uvicorn server instance

@app.get("/callback")
async def kite_callback(request: Request):
    """
    Endpoint that receives the callback from Kite Connect after successful login.
    Parses the request_token and shuts down the server.
    """
    global request_token_received
    global server_instance

    query_params = dict(request.query_params)

    if "request_token" in query_params:
        request_token_received = query_params["request_token"]
        print(f"\n--- Successfully received request_token! ---")
        print(f"Request Token: {request_token_received}")
        print("Closing server...")

        # Schedule server shutdown
        if server_instance:
            # Uvicorn needs to be shut down in the main thread/event loop
            # Here we're triggering it via an event.
            # ERROR: TODO: Nonetype cant be used in await expressions
            await request.app.shutdown_event.set() # Signal the shutdown event

        return HTMLResponse(content="<h1>Authentication successful! You can close this tab.</h1><p>Request Token received. Check your console.</p>", status_code=200)
    else:
        error_message = query_params.get("error_message", "Unknown error")
        error_code = query_params.get("error_code", "N/A")
        print(f"\n--- Error during Kite authentication ---")
        print(f"Error Code: {error_code}")
        print(f"Error Message: {error_message}")

        # You might want to shut down or keep the server running, depending on desired behavior
        if server_instance:
            await request.app.shutdown_event.set() # Still shut down if an error occurred

        return HTMLResponse(content=f"<h1>Authentication failed!</h1><p>Error: {error_message}</p><p>Check your console for details.</p>", status_code=400)


@app.on_event("startup")
async def startup_event():
    # Create an asyncio event to signal server shutdown
    app.shutdown_event = asyncio.Event()

@app.on_event("shutdown")
async def shutdown_event():
    # Wait for the shutdown event to be set
    await app.shutdown_event.wait()
    # Perform cleanup if any
    print("Server shutting down.")


def get_kite_login_url(api_key) -> str:
    """
    Generate Kite Connect login URL for getting request token

    Args:
        api_key (str): Your Kite Connect API key

    Returns:
        str: Login URL
    """
    try:
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=api_key)
        return kite.login_url()
    except ImportError:
        logging.error("KiteConnect not installed. Install with: pip install kiteconnect")
        return ""



# --- Main Logic ---
async def kite_init(api_key, api_secret, callback_url):
    global request_token_received
    global server_instance

    # 1. Get login URL
    login_url = get_kite_login_url(api_key)
    print(f"Visit this URL to login: {login_url}")
    login_url += "&redirect_url={}".format(callback_url)

    # 2. Open URL automatically
    print("\nOpening login URL in your default browser...")
    try:
        webbrowser.open_new_tab(login_url)
    except Exception as e:
        print(f"Failed to open browser automatically: {e}")
        print("Please copy and paste the URL into your browser manually.")
        print(login_url)

    # 3. Start FastAPI server in a separate thread/process or manage its lifecycle
    # We need to run uvicorn programmatically.
    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="info")
    server_instance = uvicorn.Server(config)

    # We need to run the server in a separate thread so the main thread can wait for a result.
    # Alternatively, we can use asyncio.gather if 'main' is already an async function
    # that we're running with asyncio.run().

    # Run server in the background (using a threading.Thread)
    server_thread = threading.Thread(target=asyncio.run, args=(server_instance.serve(),))
    server_thread.start()

    print(f"FastAPI server listening on {callback_url}...")
    print("Waiting for Kite callback...")

    # Wait for the request_token to be received or server to shut down
    while request_token_received is None and server_thread.is_alive():
        # Check if the server has been explicitly told to shut down (e.g., via app.shutdown_event)
        if server_instance.should_exit: # Uvicorn's internal flag
            break
        await asyncio.sleep(1) # Don't busy-wait

    # After loop, server_thread should ideally be ending or have ended.
    # It might take a moment for the thread to fully close after signal.
    # We've set up the FastAPI server to shut down via an event from the callback.
    # The main thread just needs to wait for the result.

    print("Server stopped.")

    if request_token_received:
        print("\nProceeding to generate access token (conceptual)...")

        kite = KiteConnect(api_key=api_key)
        session = kite.generate_session(request_token_received, api_secret=api_secret)
        access_token = session['access_token']

        kite.set_access_token(access_token)

        # Test the connection
        profile = kite.profile()
        print(profile)

        return kite
    else:
        print("Failed to receive request token.")
        return None

if __name__ == "__main__":
    # Ensure that `uvicorn.Server.serve()` is awaited within an asyncio event loop.
    # The `asyncio.run()` function is the correct way to run the top-level async function.
    api_key = os.getenv("KITE_API_KEY")
    callback_url = "http://localhost:8000/callback"
    api_secret = os.getenv("KITE_API_SECRET")
    asyncio.run(kite_init(api_key, api_secret, callback_url))
