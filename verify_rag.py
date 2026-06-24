import os
import sys
import time
import subprocess
import json
import urllib.request
import urllib.error
# pyrefly: ignore [missing-import]
import fitz  # PyMuPDF

PORT = 5000
BASE_URL = f"http://localhost:{PORT}/api/v1"

def create_test_pdf(filename: str):
    """
    Creates a valid PDF document with specific facts using PyMuPDF.
    """
    doc = fitz.open()
    page = doc.new_page()
    
    content = (
        "Project Nebula-X is the top secret quantum computing initiative led by Team 3.\n"
        "The project is scheduled for initial prototype completion in November 2026.\n"
        "The current lead scientist is Dr. Samantha Carter, who specializes in superconducting qubits.\n"
        "Any questions regarding Project Nebula-X should be directed to room 404 in the main laboratory."
    )
    
    # Draw text on the page
    rect = fitz.Rect(50, 50, 500, 300)
    page.insert_textbox(rect, content, fontsize=11, fontname="helv")
    doc.save(filename)
    doc.close()
    print(f"Created test PDF asset: {filename}")

def send_post_json(url: str, data: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode('utf-8'))

def send_post_file(url: str, filepath: str) -> dict:
    # Build multipart/form-data boundary
    boundary = "----Boundary12345"
    filename = os.path.basename(filepath)
    
    with open(filepath, 'rb') as f:
        file_content = f.read()
        
    part_header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/pdf\r\n\r\n"
    ).encode('utf-8')
    
    part_footer = f"\r\n--{boundary}--\r\n".encode('utf-8')
    body = part_header + file_content + part_footer
    
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body))
        },
        method="POST"
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode('utf-8'))

def send_get(url: str) -> dict:
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode('utf-8'))

def send_delete(url: str) -> dict:
    req = urllib.request.Request(url, method="DELETE")
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode('utf-8'))

def send_patch_json(url: str, data: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers={"Content-Type": "application/json"},
        method="PATCH"
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode('utf-8'))

def main():
    print("=== STARTING RAG BACKEND VERIFICATION SUITE ===")
    
    # 1. Create a dummy test PDF
    pdf_filename = "test_nebula_x.pdf"
    create_test_pdf(pdf_filename)
    
    # 2. Start Flask server as a subprocess
    # Setup environment
    env = os.environ.copy()
    env["FLASK_PORT"] = str(PORT)
    env["STORAGE_DIR"] = "./storage_test"
    
    # Clean old test storage
    if os.path.exists("./storage_test"):
        import shutil
        try:
            shutil.rmtree("./storage_test")
        except Exception:
            pass
            
    print("Starting Flask app server subprocess...")
    server_log = open("server_stdout.log", "w", encoding="utf-8")
    server_err = open("server_stderr.log", "w", encoding="utf-8")
    server_process = subprocess.Popen(
        [sys.executable, "app.py"],
        env=env,
        stdout=server_log,
        stderr=server_err,
        text=True
    )
    
    # Wait for server to boot
    print("Waiting 5 seconds for server boot...")
    time.sleep(5)
    
    # Check if server started successfully
    if server_process.poll() is not None:
        print("Error: Flask server failed to start. Check server_stdout.log and server_stderr.log")
        try:
            server_log.close()
            server_err.close()
        except Exception:
            pass
        if os.path.exists(pdf_filename):
            os.remove(pdf_filename)
        sys.exit(1)
        
    try:
        # Test 1: Upload step
        print("\n--- Testing Document Upload wizard step ---")
        upload_url = f"{BASE_URL}/admin/materials/upload"
        upload_res = send_post_file(upload_url, pdf_filename)
        print("Upload response:", upload_res)
        tracking_token = upload_res.get("tracking_token")
        assert tracking_token is not None, "Failed to get tracking token."
        
        # Test 2: Finalize step
        print("\n--- Testing Document Finalize wizard step ---")
        finalize_url = f"{BASE_URL}/admin/materials/finalize"
        finalize_payload = {
            "tracking_token": tracking_token,
            "annotation": "Official documentation containing project codename Nebula-X details."
        }
        finalize_res = send_post_json(finalize_url, finalize_payload)
        print("Finalize response:", finalize_res)
        
        # Wait for background ingestion processing (poll status for up to 60 seconds)
        print("Waiting for background chunking, embedding, database, and FAISS insertion...")
        for i in range(30):
            time.sleep(2)
            try:
                list_url = f"{BASE_URL}/admin/materials?page=1&size=50"
                list_res = send_get(list_url)
                materials = list_res.get("materials", [])
                uploaded_doc = next((m for m in materials if m["id"] == tracking_token), None)
                if uploaded_doc and uploaded_doc.get("status") == "active":
                    print(f"Document activated after {2 * (i + 1)} seconds.")
                    break
            except Exception as poll_err:
                print(f"Polling warning: {poll_err}")
        
        # Test 3: Get Dashboard & Verify Status
        print("\n--- Testing Paginated Materials Listing ---")
        list_url = f"{BASE_URL}/admin/materials?page=1&size=50"
        list_res = send_get(list_url)
        print(f"Materials list: Found {list_res.get('total')} documents.")
        materials = list_res.get("materials", [])
        assert len(materials) > 0, "No materials listed."
        
        uploaded_doc = next((m for m in materials if m["id"] == tracking_token), None)
        assert uploaded_doc is not None, "Uploaded document missing from list."
        print("Uploaded document status:", uploaded_doc.get("status"))
        print("Uploaded document annotation:", uploaded_doc.get("annotation"))
        assert uploaded_doc.get("status") == "active", "Document did not activate after processing."
        
        # Test 4: RAG Query (Successful match)
        print("\n--- Testing RAG Chat Query (Match) ---")
        query_url = f"{BASE_URL}/chat/query"
        # Match test
        query_payload_match = {
            "query": "What is the codename of the quantum computing initiative by Dr. Samantha Carter?",
            "stream": False
        }
        match_res = send_post_json(query_url, query_payload_match)
        print("User Query:", query_payload_match["query"])
        print("RAG Response:", match_res.get("response"))
        assert "Nebula-X" in match_res.get("response", "") or "Nebula - X" in match_res.get("response", ""), "Query response did not contain matching context facts."
        
        # Test 5: RAG Query (Fallback / Unrelated query)
        print("\n--- Testing RAG Chat Query (Fallback) ---")
        query_payload_fallback = {
            "query": "How many states are in the United States?",
            "stream": False
        }
        fallback_res = send_post_json(query_url, query_payload_fallback)
        print("User Query:", query_payload_fallback["query"])
        print("RAG Response:", fallback_res.get("response"))
        assert fallback_res.get("response") == "I could not find relevant information in the knowledge base.", "Fallback response mismatch."
        
        # Test 6: Status toggle PATCH
        print("\n--- Testing Material Status Patch Toggle ---")
        patch_url = f"{BASE_URL}/admin/materials/{tracking_token}/status"
        patch_res = send_patch_json(patch_url, {"status": "inactive"})
        print("Patch response:", patch_res)
        
        # Verify status is inactive
        list_res2 = send_get(list_url)
        uploaded_doc2 = next((m for m in list_res2.get("materials", []) if m["id"] == tracking_token), None)
        assert uploaded_doc2 is not None, "Uploaded document missing from listing after status patch."
        assert uploaded_doc2.get("status") == "inactive", "Failed to change status to inactive."
        
        # Test 7: Verify query fails when document is inactive
        print("\n--- Testing Query Bypasses Inactive Materials ---")
        query_res_inactive = send_post_json(query_url, query_payload_match)
        print("RAG Response (with inactive document):", query_res_inactive.get("response"))
        assert query_res_inactive.get("response") == "I could not find relevant information in the knowledge base.", "Inactive material was not bypassed."
        
        # Test 8: Permanent Delete
        print("\n--- Testing Document Permanent Deletion ---")
        delete_url = f"{BASE_URL}/admin/materials/{tracking_token}"
        delete_res = send_delete(delete_url)
        print("Delete response:", delete_res)
        
        # Confirm deleted from dashboard
        list_res3 = send_get(list_url)
        uploaded_doc3 = next((m for m in list_res3.get("materials", []) if m["id"] == tracking_token), None)
        assert uploaded_doc3 is None, "Material was not deleted from database."
        print("Confirmed document is completely removed from list.")
        
        print("\nALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
        
    except AssertionError as e:
        print(f"\nTEST FAILURE: {e}")
        # Show server logs on failure
        sys.exit(1)
    except urllib.error.HTTPError as e:
        print(f"\nHTTP ERROR: {e.code} - {e.reason}")
        try:
            print("Response:", e.read().decode('utf-8'))
        except Exception:
            pass
        sys.exit(1)
    except Exception as e:
        print(f"\nUNEXPECTED FAILURE: {e}")
        sys.exit(1)
    finally:
        # Terminate server
        print("Terminating server process...")
        if 'server_process' in locals():
            try:
                server_process.terminate()
                server_process.wait()
            except Exception:
                pass
        try:
            if 'server_log' in locals():
                server_log.close()
            if 'server_err' in locals():
                server_err.close()
        except Exception:
            pass
        
        # Cleanup file
        if os.path.exists(pdf_filename):
            os.remove(pdf_filename)
        
        # Clean test storage
        if os.path.exists("./storage_test"):
            import shutil
            try:
                shutil.rmtree("./storage_test")
            except Exception:
                pass
        print("Verification script cleanup completed.")

if __name__ == "__main__":
    main()
