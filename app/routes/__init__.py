"""
app/routes/__init__.py

This package holds Flask Blueprints (controllers in our MVC layout).
Blueprints are imported and registered inside app/__init__.py's create_app().

Only main_routes.py exists in this phase. Later phases add:
  - auth_routes.py      (Phase 6: register/login/logout)
  - search_routes.py    (Phase 6/7: search by title/author/keyword/domain)
  - paper_routes.py     (Phase 7: paper detail + similar papers)
  - dashboard_routes.py (Phase 7: personalized dashboard)
  - bookmark_routes.py  (Phase 7: bookmark CRUD)
  - admin_routes.py     (Phase 7: admin paper management)
"""
