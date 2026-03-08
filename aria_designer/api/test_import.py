import sys
sys.path.insert(0, ".")
try:
    from app.routers.workflows import router
    from app.routers.components import router
    print("SUCCESS")
except Exception as e:
    print("ERROR:", e)
