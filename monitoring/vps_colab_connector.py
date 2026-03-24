import requests
import json
import os
import re # Import regex module

def send_data_to_colab(
    endpoint: str,
    data: dict,
    colab_api_key: str,
    colab_url_file_path: str,
    log_func=print
) -> bool:
    """
    Sends data to the Colab API receiver.

    Args:
        endpoint (str): The specific endpoint on the Colab Flask app (e.g., "update_monitor_data").
        data (dict): The dictionary containing the data to be sent.
        colab_api_key (str): The API key for authenticating with the Colab Flask app.
        colab_url_file_path (str): The file path on the VPS where the Colab public URL is stored.
        log_func (callable): Function to use for logging (e.g., print, or a custom logger).

    Returns:
        bool: True if transfer was successful, False otherwise.
    """
    try:
        # Read the public URL of the Colab API from file
        if not os.path.exists(colab_url_file_path):
            log_func(f"[ERROR] Colab public URL file not found at {colab_url_file_path}. Cannot send data.")
            return False

        with open(colab_url_file_path, 'r') as f:
            public_url_raw = f.read().strip()

        if not public_url_raw:
            log_func("[ERROR] Colab public URL is empty. Cannot send data.")
            return False

        # Extract the actual URL from the NgrokTunnel string representation if present
        url_match = re.search(r'(https?://[^\s"]+)', public_url_raw)
        if url_match:
            public_url = url_match.group(1)
        else:
            # If it's not the NgrokTunnel string, assume it's already a clean URL
            public_url = public_url_raw
        
        if not public_url.startswith('http'):
            log_func(f"[ERROR] Extracted URL '{public_url}' is not a valid HTTP/HTTPS URL. Cannot send data.")
            return False

        headers = {
            "Content-Type": "application/json",
            "X-API-Key": colab_api_key # Use the API key defined in Colab
        }

        # --- SOLUSI TANGGUH: PRIORITAS LOKAL ---
        # Kita abaikan Ngrok dulu karena Bot & Flask ada di VPS yang sama
        local_port = 5000 
        target_url = f"http://127.0.0.1:{local_port}/{endpoint}"

        try:
            response = requests.post(target_url, headers=headers, json=data, timeout=5)
        except requests.exceptions.ConnectionError:
            # Jika localhost gagal, baru coba pakai URL dari Ngrok (public_url)
            base_url = public_url.rstrip('/')
            target_url = f"{base_url}/{endpoint}"
            response = requests.post(target_url, headers=headers, json=data, timeout=10)
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

        log_func(f"[INFO] Successfully sent data to Colab endpoint {endpoint}. Status Code: {response.status_code}")
        return True
    except requests.exceptions.Timeout:
        log_func(f"[ERROR] Request to Colab endpoint {endpoint} timed out.")
    except requests.exceptions.ConnectionError as e:
        log_func(f"[ERROR] Connection error to Colab endpoint {endpoint}: {e}. Is Colab notebook running and ngrok active?")
    except requests.exceptions.HTTPError as e:
        log_func(f"[ERROR] HTTP error during transfer to Colab endpoint {endpoint}: {e}. Response: {response.text}")
    except Exception as e:
        log_func(f"[ERROR] An unexpected error occurred while sending data to Colab: {e}")
    return False
