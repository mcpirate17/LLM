
def generate_dispatchers():
    return """
def dispatch_exp_map(cnp.ndarray[float, ndim=1] x, float c):
    cdef int64_t n = x.shape[0]
    cdef cnp.ndarray[float, ndim=1] y = np.empty(n, dtype=np.float32)
    aria_exp_map_f32(<float*>x.data, <float*>y.data, n, c)
    return y

def dispatch_log_map(cnp.ndarray[float, ndim=1] x, float c):
    cdef int64_t n = x.shape[0]
    cdef cnp.ndarray[float, ndim=1] y = np.empty(n, dtype=np.float32)
    aria_log_map_f32(<float*>x.data, <float*>y.data, n, c)
    return y

def dispatch_poincare_add(cnp.ndarray[float, ndim=2] x, cnp.ndarray[float, ndim=2] v, float c):
    cdef int64_t batch = x.shape[0]
    cdef int64_t dim = x.shape[1]
    cdef cnp.ndarray[float, ndim=2] y = np.empty((batch, dim), dtype=np.float32)
    aria_poincare_add_f32(<float*>x.data, <float*>v.data, <float*>y.data, batch, dim, c)
    return y

def dispatch_hyp_linear(cnp.ndarray[float, ndim=2] x, cnp.ndarray[float, ndim=2] W, float c):
    cdef int64_t batch = x.shape[0]
    cdef int64_t dim_in = x.shape[1]
    cdef int64_t dim_out = W.shape[0]
    cdef cnp.ndarray[float, ndim=2] y = np.empty((batch, dim_out), dtype=np.float32)
    aria_hyp_linear_f32(<float*>x.data, <float*>W.data, <float*>y.data, batch, dim_in, dim_out, c)
    return y

def dispatch_hyperbolic_norm(cnp.ndarray[float, ndim=2] x, cnp.ndarray[float, ndim=1] gamma, cnp.ndarray[float, ndim=1] beta, float c, float eps=1e-5):
    cdef int64_t batch = x.shape[0]
    cdef int64_t dim = x.shape[1]
    cdef cnp.ndarray[float, ndim=2] y = np.empty((batch, dim), dtype=np.float32)
    cdef float* beta_ptr = NULL
    if beta is not None:
        beta_ptr = <float*>beta.data
    aria_hyperbolic_norm_f32(<float*>x.data, <float*>gamma.data, beta_ptr, <float*>y.data, batch, dim, c, eps)
    return y

def dispatch_hyp_tangent_nonlinear(cnp.ndarray[float, ndim=1] x, float c):
    cdef int64_t n = x.shape[0]
    cdef cnp.ndarray[float, ndim=1] y = np.empty(n, dtype=np.float32)
    aria_hyp_tangent_nonlinear_f32(<float*>x.data, <float*>y.data, n, c)
    return y

def dispatch_tropical_attention(cnp.ndarray[float, ndim=3] x, float temperature=1.0):
    cdef int64_t batch = x.shape[0]
    cdef int64_t seq = x.shape[1]
    cdef int64_t dim = x.shape[2]
    cdef cnp.ndarray[float, ndim=3] y = np.empty((batch, seq, dim), dtype=np.float32)
    aria_tropical_attention_f32(<float*>x.data, <float*>y.data, batch, seq, dim, temperature)
    return y

def dispatch_tropical_center(cnp.ndarray[float, ndim=3] x):
    cdef int64_t batch = x.shape[0]
    cdef int64_t seq = x.shape[1]
    cdef int64_t dim = x.shape[2]
    cdef cnp.ndarray[float, ndim=3] y = np.empty((batch, seq, dim), dtype=np.float32)
    aria_tropical_center_f32(<float*>x.data, <float*>y.data, batch, seq, dim)
    return y

def dispatch_tropical_gate(cnp.ndarray[float, ndim=3] x, float temperature=1.0):
    cdef int64_t batch = x.shape[0]
    cdef int64_t seq = x.shape[1]
    cdef int64_t dim = x.shape[2]
    cdef cnp.ndarray[float, ndim=3] y = np.empty((batch, seq, dim), dtype=np.float32)
    aria_tropical_gate_f32(<float*>x.data, <float*>y.data, batch, seq, dim, temperature)
    return y

def dispatch_tropical_add(cnp.ndarray[float, ndim=1] a, cnp.ndarray[float, ndim=1] b):
    cdef int64_t n = a.shape[0]
    cdef cnp.ndarray[float, ndim=1] y = np.empty(n, dtype=np.float32)
    aria_tropical_add_f32(<float*>a.data, <float*>b.data, <float*>y.data, n)
    return y

def dispatch_tropical_matmul(cnp.ndarray[float, ndim=2] A, cnp.ndarray[float, ndim=2] B):
    cdef int64_t M = A.shape[0]
    cdef int64_t K = A.shape[1]
    cdef int64_t N = B.shape[1]
    cdef cnp.ndarray[float, ndim=2] C = np.empty((M, N), dtype=np.float32)
    aria_tropical_matmul_f32(<float*>A.data, <float*>B.data, <float*>C.data, M, K, N)
    return C

def dispatch_padic_gate(cnp.ndarray[float, ndim=1] x, float p=2.0):
    cdef int64_t n = x.shape[0]
    cdef cnp.ndarray[float, ndim=1] y = np.empty(n, dtype=np.float32)
    aria_padic_gate_f32(<float*>x.data, <float*>y.data, n, p)
    return y

def dispatch_padic_expand(cnp.ndarray[float, ndim=2] x, cnp.ndarray[float, ndim=2] W, float p=2.0, int64_t n_digits=3):
    cdef int64_t batch = x.shape[0]
    cdef int64_t dim_ext = W.shape[0]
    cdef cnp.ndarray[float, ndim=2] y = np.empty((batch, dim_ext), dtype=np.float32)
    aria_padic_expand_f32(<float*>x.data, <float*>W.data, <float*>y.data, batch, x.shape[1], p, n_digits)
    return y

def dispatch_padic_residual(cnp.ndarray[float, ndim=2] x, cnp.ndarray[float, ndim=2] W, float p=2.0, int64_t n_digits=3):
    cdef int64_t batch = x.shape[0]
    cdef int64_t dim = x.shape[1]
    cdef cnp.ndarray[float, ndim=2] y = np.empty((batch, dim), dtype=np.float32)
    aria_padic_residual_f32(<float*>x.data, <float*>W.data, <float*>y.data, batch, dim, p, n_digits)
    return y

def dispatch_ultrametric_attention(cnp.ndarray[float, ndim=3] x, float p=2.0):
    cdef int64_t batch = x.shape[0]
    cdef int64_t seq = x.shape[1]
    cdef int64_t dim = x.shape[2]
    cdef cnp.ndarray[float, ndim=3] y = np.empty((batch, seq, dim), dtype=np.float32)
    aria_ultrametric_attention_f32(<float*>x.data, <float*>y.data, batch, seq, dim, p)
    return y

def dispatch_rotor_transform(cnp.ndarray[float, ndim=2] x, cnp.ndarray[float, ndim=1] rotor):
    cdef int64_t batch = x.shape[0]
    cdef int64_t dim = x.shape[1]
    cdef cnp.ndarray[float, ndim=2] y = np.empty((batch, dim), dtype=np.float32)
    aria_rotor_transform_f32(<float*>x.data, <float*>rotor.data, <float*>y.data, batch, dim)
    return y

def dispatch_grade_select(cnp.ndarray[float, ndim=2] x, int32_t grade):
    cdef int64_t batch = x.shape[0]
    cdef int64_t dim = x.shape[1]
    cdef cnp.ndarray[float, ndim=2] y = np.empty((batch, dim), dtype=np.float32)
    aria_grade_select_f32(<float*>x.data, <float*>y.data, batch, dim, grade)
    return y

def dispatch_grade_mix(cnp.ndarray[float, ndim=2] x, cnp.ndarray[float, ndim=1] alpha):
    cdef int64_t batch = x.shape[0]
    cdef int64_t dim = x.shape[1]
    cdef cnp.ndarray[float, ndim=2] y = np.empty((batch, dim), dtype=np.float32)
    aria_grade_mix_f32(<float*>x.data, <float*>alpha.data, <float*>y.data, batch, dim)
    return y

def dispatch_clifford_attention(cnp.ndarray[float, ndim=3] x):
    cdef int64_t batch = x.shape[0]
    cdef int64_t seq = x.shape[1]
    cdef int64_t dim = x.shape[2]
    cdef cnp.ndarray[float, ndim=3] y = np.empty((batch, seq, dim), dtype=np.float32)
    aria_clifford_attention_f32(<float*>x.data, <float*>y.data, batch, seq, dim)
    return y

"""

with open("/home/tim/Projects/LLM/research/runtime/native/cython/aria_bridge_additions.pyx", "w") as f:
    f.write(generate_dispatchers())

print("Done")
