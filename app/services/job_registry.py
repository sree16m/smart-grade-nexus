from typing import Dict, Any, Optional, List
import time

class JobRegistry:
    def __init__(self):
        # schema: { "book_name": { "status": str, "current_page": int, "total_pages": int, "cancelled": bool, "last_updated": float } }
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def start_job(self, book_name: str, total_pages: int):
        self._jobs[book_name] = {
            "status": "processing",
            "current_page": 0,
            "total_pages": total_pages,
            "cancelled": False,
            "last_updated": time.time()
        }

    def update_progress(self, book_name: str, current_page: int):
        if book_name in self._jobs:
            self._jobs[book_name].update({
                "current_page": current_page,
                "last_updated": time.time()
            })

    def complete_job(self, book_name: str):
        if book_name in self._jobs:
            self._jobs[book_name].update({
                "status": "completed",
                "last_updated": time.time()
            })

    def fail_job(self, book_name: str, error: str):
        if book_name in self._jobs:
            self._jobs[book_name].update({
                "status": "failed",
                "error": error,
                "last_updated": time.time()
            })

    def cancel_job(self, book_name: str):
        if book_name in self._jobs:
            self._jobs[book_name].update({
                "cancelled": True,
                "status": "cancelling",
                "last_updated": time.time()
            })
            return True
        return False

    def is_cancelled(self, book_name: str) -> bool:
        return self._jobs.get(book_name, {}).get("cancelled", False)

    def get_status(self, book_name: str) -> Optional[Dict[str, Any]]:
        return self._jobs.get(book_name)

    def list_jobs(self) -> List[Dict[str, Any]]:
        return [{"book_name": name, **data} for name, data in self._jobs.items()]

# Singleton Instance
job_registry = JobRegistry()
