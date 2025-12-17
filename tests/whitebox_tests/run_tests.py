import subprocess
import os
import sys
import time
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler

# Global server variable to handle cleanup
httpd = None

# Force UTF-8 for stdout to handle emojis in tests on Windows
sys.stdout.reconfigure(encoding='utf-8')

def start_server(root_dir, port=8000):
    global httpd
    print(f"Starting test server at port {port} serving {root_dir}")
    os.chdir(root_dir)
    server_address = ('', port)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    httpd.serve_forever()

def run_tests():
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__)) # tests/whitebox_tests
    tests_dir = os.path.dirname(current_dir) # tests
    project_root = os.path.dirname(tests_dir) # humanorai
    frontend_dir = os.path.join(project_root, "app", "frontend")
    
    # Start Server in a separate thread
    port = 8000
    server_thread = threading.Thread(target=start_server, args=(frontend_dir, port))
    server_thread.daemon = True
    server_thread.start()
    
    # Give server a moment to start
    time.sleep(2)
    
    # Set URL for tests
    test_url = f"http://localhost:{port}/index.html"
    os.environ["TEST_BASE_URL"] = test_url
    
    test_files = [
        "test_modal.py",
        "test_input_validation.py",
        "test_file_upload_ui.py"
    ]

    print(f"Running whitebox tests against: {test_url}")
    print("-" * 50)

    results = {}

    try:
        for test_file in test_files:
            print(f"\n[RUNNING] {test_file}...")
            file_path = os.path.join(current_dir, test_file)
            
            try:
                # Run the test file as a separate process
                # Pass current env with TEST_BASE_URL and force UTF-8 for IO
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"
                result = subprocess.run(
                    [sys.executable, file_path],
                    capture_output=True,
                    text=True,
                    check=False,
                    env=env,
                    encoding="utf-8",
                    errors="replace"
                )
                
                # Print output
                print(result.stdout)
                if result.stderr:
                    print("Errors/Warnings:")
                    print(result.stderr)
                
                if "[Fail]" in result.stdout or "[Error]" in result.stdout or result.returncode != 0:
                    results[test_file] = "FAILED"
                else:
                    results[test_file] = "PASSED"

            except Exception as e:
                print(f"Exception running {test_file}: {e}")
                results[test_file] = "ERROR"
    finally:
        # Stop server (daemon thread will die when main process exits, 
        # but explicit shutdown is cleaner if we were staying alive)
        if httpd:
            httpd.shutdown()
            print("Server stopped.")

    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    for test, status in results.items():
        print(f"{test:<30} : {status}")
    print("=" * 50)

if __name__ == "__main__":
    run_tests()
