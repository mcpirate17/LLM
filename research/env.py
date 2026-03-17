try:
    import aria_core

    HAS_ARIA_CORE = True
except ImportError:
    aria_core = None
    HAS_ARIA_CORE = False
