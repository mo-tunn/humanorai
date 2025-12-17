import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
# Removed invalid import

def test_input_validation():
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
        time.sleep(1)

        # 1. Enter short text
        article_input = driver.find_element(By.ID, "article-input")
        short_text = "Short text"
        article_input.send_keys(short_text)
        print(f"[Info] Entered text: '{short_text}' (len: {len(short_text)})")

        # 2. Click Analyze
        analyze_btn = driver.find_element(By.ID, "analyze-button")
        # Use JS click
        driver.execute_script("arguments[0].click();", analyze_btn)
        print("[Info] Clicked Analyze button")

        # 3. Wait for Alert and Check text
        try:
            WebDriverWait(driver, 5).until(EC.alert_is_present())
            alert = driver.switch_to.alert
            alert_text = alert.text
            expected_text = "Minimum 50 characters required for analysis."
            
            if alert_text == expected_text:
                print(f"[Pass] Alert text matches: '{alert_text}'")
            else:
                print(f"[Fail] Alert text mismatch. Got: '{alert_text}', Expected: '{expected_text}'")
            
            alert.accept()

        except Exception as e:
            print(f"[Fail] Alert did not appear or other error: {e}")

    except Exception as e:
        print(f"[Error] Test Failed: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    test_input_validation()
