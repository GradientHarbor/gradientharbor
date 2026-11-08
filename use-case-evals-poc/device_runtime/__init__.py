"""The phone's on-device stack, served over HTTP so the iOS Simulator app can use it.

MLX doesn't run in the iOS Simulator, so for the Simulator demo the "phone" side (fine-tuned
student, router, redactor, tools over the member's local data, privacy ledger) runs here on the
Mac: the same code and weights the evals measured. The cloud side stays separate: escalations go
to the backend's teacher proxy (backend/, port 8000), exactly as they would from a real iPhone.
"""
