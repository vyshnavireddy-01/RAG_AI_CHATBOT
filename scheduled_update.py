# scheduled_update.py
# This runs independently — Flask doesn't need to be running

import sys
import os
import logging

# Setup logging to show in terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from auto_updater import check_and_update

if __name__ == "__main__":
    print("🔍 Starting Ealkay sitemap check...")
    check_and_update()
    print("✅ Done!")