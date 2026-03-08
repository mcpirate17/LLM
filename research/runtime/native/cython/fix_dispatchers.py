
with open("/home/tim/Projects/LLM/research/runtime/native/cython/aria_bridge.pyx", "r") as f:
    text = f.read()

text = text.replace("def dispatch_exp_map(cnp.ndarray[float, ndim=1] x, float c):\n    cdef int64_t n = x.shape[0]\n    cdef cnp.ndarray[float, ndim=1] y = np.empty(n, dtype=np.float32)\n    aria_exp_map_f32(<float*>x.data, <float*>y.data, n, c)\n    return y", "def dispatch_exp_map(cnp.ndarray[float, ndim=2] x, float c):\n    cdef int64_t batch = x.shape[0]\n    cdef int64_t dim = x.shape[1]\n    cdef cnp.ndarray[float, ndim=2] y = np.empty((batch, dim), dtype=np.float32)\n    aria_exp_map_f32(<float*>x.data, <float*>y.data, batch, dim, c)\n    return y")

text = text.replace("def dispatch_log_map(cnp.ndarray[float, ndim=1] x, float c):\n    cdef int64_t n = x.shape[0]\n    cdef cnp.ndarray[float, ndim=1] y = np.empty(n, dtype=np.float32)\n    aria_log_map_f32(<float*>x.data, <float*>y.data, n, c)\n    return y", "def dispatch_log_map(cnp.ndarray[float, ndim=2] x, float c):\n    cdef int64_t batch = x.shape[0]\n    cdef int64_t dim = x.shape[1]\n    cdef cnp.ndarray[float, ndim=2] y = np.empty((batch, dim), dtype=np.float32)\n    aria_log_map_f32(<float*>x.data, <float*>y.data, batch, dim, c)\n    return y")

with open("/home/tim/Projects/LLM/research/runtime/native/cython/aria_bridge.pyx", "w") as f:
    f.write(text)

