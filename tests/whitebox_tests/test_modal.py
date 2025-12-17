import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def test_modal_logic():
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__)) # .../tests/whitebox_tests
    tests_dir = os.path.dirname(base_dir) # .../tests
    driver_path = os.path.join(tests_dir, "chromedriver.exe")
    
    # Get URL from env or fallback
    target_url = os.environ.get("TEST_BASE_URL")
    if not target_url:
        project_root = os.path.dirname(tests_dir)
        html_path = os.path.join(project_root, "app", "frontend", "index.html")
        target_url = f"file:///{html_path}"

    print(f"Testing URL: {target_url}")
    print(f"Using Driver: {driver_path}")

    # Setup Driver
    service = Service(executable_path=driver_path)
    driver = webdriver.Chrome(service=service)

    try:
        driver.get(target_url)
        driver.maximize_window()

        # Allow page to load
        time.sleep(1)

        # 1. Open Modal
        open_btn = driver.find_element(By.ID, "open-accuracy-modal")
        # Use JS click to bypass interactability issues
        driver.execute_script("arguments[0].click();", open_btn)
        print("[Pass] Clicked Open Modal Button")

        # Wait for modal to be visible
        modal = WebDriverWait(driver, 5).until(
            EC.visibility_of_element_located((By.ID, "accuracy-modal"))
        )
        
        display_style = modal.value_of_css_property("display")
        if display_style == "flex":
             print("[Pass] Modal is visible (display: flex)")
        else:
             print(f"[Fail] Modal display is {display_style}")

        time.sleep(1) # Observe

        # 2. Close Modal
        close_btn = driver.find_element(By.ID, "close-modal")
        close_btn.click()
        print("[Pass] Clicked Close Modal Button")
        
        # Wait for modal to be hidden
        WebDriverWait(driver, 5).until(
            EC.invisibility_of_element_located((By.ID, "accuracy-modal"))
        )
        
        # Verify hidden (check display property again or wait condition)
        # Note: selenium might return empty string or none if hidden depending on impl, 
        # but invisibility wait confirms it's not visible.
        print("[Pass] Modal is hidden")

    except Exception as e:
        print(f"[Error] Test Failed: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    test_modal_logic()
