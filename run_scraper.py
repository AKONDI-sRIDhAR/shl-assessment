"""
Wrapper to run the scraper with proper stdout encoding for Windows.
Saves all output to scraper_log.txt.
"""
import sys
import io
import os

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Also redirect to file
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scraper_log.txt")
log_file = open(log_path, "w", encoding="utf-8")

class TeeWriter:
    def __init__(self, *writers):
        self.writers = writers
    def write(self, s):
        for w in self.writers:
            w.write(s)
            w.flush()
    def flush(self):
        for w in self.writers:
            w.flush()

sys.stdout = TeeWriter(sys.stdout, log_file)
sys.stderr = TeeWriter(sys.stderr, log_file)

# Now run the scraper
from scraper import main
main()
log_file.close()
