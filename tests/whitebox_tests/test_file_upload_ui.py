import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def test_file_upload_ui():
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__)) # .../tests/whitebox_tests
    tests_dir = os.path.dirname(base_dir) # .../tests
    driver_path = os.path.join(tests_dir, "chromedriver.exe")
    
    # Create dummy file for testing
    dummy_file_path = os.path.join(base_dir, "dummy_test_file.txt")
    with open(dummy_file_path, "w") as f:
        f.write("This is a dummy file content for selenium testing. It has enough characters.")

    # Get URL from env or fallback
    target_url = os.environ.get("TEST_BASE_URL")
    if not target_url:
        project_root = os.path.dirname(tests_dir)
        html_path = os.path.join(project_root, "app", "frontend", "index.html")
        target_url = f"file:///{html_path}"

    print(f"Testing URL: {target_url}")
    print(f"Using Driver: {driver_path}")
    print(f"Dummy File: {dummy_file_path}")

    # Setup Driver
    service = Service(executable_path=driver_path)
    driver = webdriver.Chrome(service=service)

    try:
        driver.get(target_url)
        time.sleep(1)

        # 1. Upload File (Send keys to hidden input)
        file_input = driver.find_element(By.ID, "file-input")
        # Make it fully visible for Selenium
        driver.execute_script("arguments[0].style.display = 'block'; arguments[0].style.visibility = 'visible'; arguments[0].style.opacity = '1'; arguments[0].style.width = '100px'; arguments[0].style.height = '100px'; arguments[0].style.position = 'absolute'; arguments[0].style.top = '0'; arguments[0].style.left = '0'; arguments[0].style.zIndex = '9999';", file_input)
        
        file_input.send_keys(dummy_file_path)
        
        # Check value
        val = file_input.get_attribute("value")
        print(f"[Info] Input value after send_keys: {val}")

        # Manually trigger the change event
        driver.execute_script("arguments[0].dispatchEvent(new Event('change', { bubbles: true }));", file_input)
        
        print("[Info] Sent file path to input and dispatched change")
        time.sleep(1) # Wait for JS to process

        # 2. Verify Article Input is Disabled
        article_input = driver.find_element(By.ID, "article-input")
        is_disabled = article_input.get_attribute("disabled")
        placeholder = article_input.get_attribute("placeholder")
        
        if is_disabled:
            print("[Pass] Article input is disabled")
        else:
            print("[Fail] Article input is NOT disabled")
            
        if "File selected" in placeholder:
             # Encode to avoid charmap errors with emoji
             safe_placeholder = placeholder.encode('utf-8', 'ignore').decode('utf-8')
             print(f"[Pass] Placeholder text updated: '{safe_placeholder}'")
        else:
             safe_placeholder = placeholder.encode('utf-8', 'ignore').decode('utf-8')
             print(f"[Fail] Placeholder mismatch: '{safe_placeholder}'")

        time.sleep(1)

        # 3. Remove File
        remove_btn = driver.find_element(By.ID, "remove-file")
        # Ensure visible/clickable - might need to check if file-info is visible
        WebDriverWait(driver, 2).until(EC.visibility_of(remove_btn))
        remove_btn.click()
        print("[Info] Clicked Remove File button")

        # 4. Verify Article Input is Enabled
        time.sleep(0.5)
        is_disabled = article_input.get_attribute("disabled")
        placeholder = article_input.get_attribute("placeholder")

        if not is_disabled:
            print("[Pass] Article input is enabled again")
        else:
            print("[Fail] Article input is still disabled")
        
        if "Paste the article" in placeholder:
            print(f"[Pass] Placeholder restored: '{placeholder}'")
        else:
            print(f"[Fail] Placeholder not restored: '{placeholder}'")

    except Exception as e:
        print(f"[Error] Test Failed: {e}")
    finally:
        driver.quit()
        # Cleanup dummy file
        if os.path.exists(dummy_file_path):
            os.remove(dummy_file_path)

if __name__ == "__main__":
    test_file_upload_ui()
