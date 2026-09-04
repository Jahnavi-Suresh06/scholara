"""
app/services/__init__.py

Business logic layer (the "M" in our MVC-ish layout, alongside models).
Routes (controllers) call into these services rather than querying the
database directly -- this keeps routes thin, makes logic reusable/testable,
and is where Phase 8's recommendation logic will plug in without touching
route code.
"""
