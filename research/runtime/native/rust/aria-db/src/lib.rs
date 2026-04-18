//! aria_db — Native SQLite connection manager for LabNotebook.
//!
//! Holds a single DELETE-journal-mode connection that **never closes**,
//! with an exclusive advisory flock preventing any second writer process.
//!
//! All reads and writes go through this one connection, serialized by a
//! Rust mutex.  DELETE journal mode is used instead of WAL because the
//! single-connection architecture has no concurrent reader/writer
//! benefit from WAL, and WAL's -wal/-shm sidecar files are fatally
//! fragile under multi-process open/close (see the 2026-04-16/17
//! incident history in the pragma comment block).
//!
//! Async writes are supported via an internal crossbeam channel and a
//! dedicated writer thread, matching LabNotebook's `_submit_write` /
//! `flush_writes` API.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crossbeam_channel::{bounded, Sender};
use fs2::FileExt;
use parking_lot::Mutex;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::ToPyObject;
use rusqlite::functions::FunctionFlags;
use rusqlite::{params_from_iter, types::Value, Connection, OpenFlags};

/// Python-compatible SQLite value.
#[derive(Debug, Clone)]
enum SqliteValue {
    Null,
    Integer(i64),
    Real(f64),
    Text(String),
    Blob(Vec<u8>),
}

impl SqliteValue {
    fn to_python(&self, py: Python<'_>) -> PyObject {
        match self {
            SqliteValue::Null => py.None(),
            SqliteValue::Integer(v) => v.to_object(py),
            SqliteValue::Real(v) => v.to_object(py),
            SqliteValue::Text(v) => v.to_object(py),
            SqliteValue::Blob(v) => {
                // Return as Python `bytes`, not `list[int]`.
                pyo3::types::PyBytes::new_bound(py, v).into()
            }
        }
    }
}

fn rusqlite_value_to_ours(v: &rusqlite::types::ValueRef<'_>) -> SqliteValue {
    match v {
        rusqlite::types::ValueRef::Null => SqliteValue::Null,
        rusqlite::types::ValueRef::Integer(i) => SqliteValue::Integer(*i),
        rusqlite::types::ValueRef::Real(f) => SqliteValue::Real(*f),
        rusqlite::types::ValueRef::Text(s) => {
            SqliteValue::Text(String::from_utf8_lossy(s).to_string())
        }
        rusqlite::types::ValueRef::Blob(b) => SqliteValue::Blob(b.to_vec()),
    }
}

/// Convert a Python value to a rusqlite Value.
fn py_to_rusqlite(obj: &Bound<'_, pyo3::PyAny>) -> Result<Value, PyErr> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(v) = obj.extract::<bool>() {
        return Ok(Value::Integer(v as i64));
    }
    if let Ok(v) = obj.extract::<i64>() {
        return Ok(Value::Integer(v));
    }
    if let Ok(v) = obj.extract::<f64>() {
        return Ok(Value::Real(v));
    }
    if let Ok(v) = obj.extract::<String>() {
        return Ok(Value::Text(v));
    }
    if let Ok(v) = obj.extract::<Vec<u8>>() {
        return Ok(Value::Blob(v));
    }
    Err(PyValueError::new_err(format!(
        "Cannot convert Python value to SQLite: {:?}",
        obj
    )))
}

fn py_params(params: &Bound<'_, pyo3::PyAny>) -> Result<Vec<Value>, PyErr> {
    if params.is_none() {
        return Ok(vec![]);
    }
    let seq: Vec<Bound<'_, pyo3::PyAny>> = params.extract()?;
    seq.iter().map(py_to_rusqlite).collect()
}

/// Message types for the async writer thread.
enum WriteMsg {
    /// Execute a single statement with params.
    Execute {
        sql: String,
        params: Vec<Value>,
    },
    /// Execute a statement with multiple parameter sets (executemany).
    ExecuteMany {
        sql: String,
        params_list: Vec<Vec<Value>>,
    },
    /// Flush: commit and signal the caller.
    Flush {
        done: Arc<(parking_lot::Mutex<bool>, parking_lot::Condvar)>,
    },
    /// Shut down the writer thread.
    Stop,
}

/// Inner state holding the SQLite connection.
struct DbInner {
    conn: Connection,
}

/// The native connection manager exposed to Python.
///
/// Thread-safe: all access serialized by a Rust mutex.
/// The connection is opened once with WAL mode and never closed.
///
/// In read-write mode, holds an exclusive advisory file lock on
/// ``<db_path>.writer-lock`` for the lifetime of the process. This prevents
/// a second Python process from opening the same database for writing and
/// triggering the orphan-WAL failure mode on its exit (SQLite's close
/// teardown can unlink the WAL file out from under a long-running writer,
/// stranding hours of in-flight writes under a held file descriptor that
/// points at a deleted inode). The kernel auto-releases the flock on
/// process exit, including crashes, so stale locks cannot wedge restart.
///
/// Tests, scripts, and tools that only need to query should use
/// ``get_manager_readonly()`` instead; read-only connections do not take
/// the writer lock and cannot trigger the teardown pathway.
#[pyclass]
struct ConnectionManager {
    inner: Arc<Mutex<DbInner>>,
    writer_tx: Mutex<Option<Sender<WriteMsg>>>,
    writer_started: AtomicBool,
    db_path: String,
    read_only: bool,
    // Held for the lifetime of the manager. Dropping closes the lockfile and
    // releases the flock. We store it in a Mutex<Option<File>> so the Drop
    // impl can be explicit about the release order (lock first, conn later).
    writer_lock: Mutex<Option<File>>,
}

/// Register math functions that Python's sqlite3 provides but rusqlite's
/// bundled SQLite does not (it lacks SQLITE_ENABLE_MATH_FUNCTIONS).
fn register_math_functions(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.create_scalar_function(
        "sqrt",
        1,
        FunctionFlags::SQLITE_UTF8 | FunctionFlags::SQLITE_DETERMINISTIC,
        |ctx| {
            let val = ctx.get_raw(0);
            match val {
                rusqlite::types::ValueRef::Null => Ok(None::<f64>),
                rusqlite::types::ValueRef::Integer(i) => Ok(Some((i as f64).sqrt())),
                rusqlite::types::ValueRef::Real(f) => Ok(Some(f.sqrt())),
                _ => Ok(None),
            }
        },
    )?;
    conn.create_scalar_function(
        "log",
        1,
        FunctionFlags::SQLITE_UTF8 | FunctionFlags::SQLITE_DETERMINISTIC,
        |ctx| {
            let val = ctx.get_raw(0);
            match val {
                rusqlite::types::ValueRef::Null => Ok(None::<f64>),
                rusqlite::types::ValueRef::Integer(i) => Ok(Some((i as f64).ln())),
                rusqlite::types::ValueRef::Real(f) => Ok(Some(f.ln())),
                _ => Ok(None),
            }
        },
    )?;
    conn.create_scalar_function(
        "exp",
        1,
        FunctionFlags::SQLITE_UTF8 | FunctionFlags::SQLITE_DETERMINISTIC,
        |ctx| {
            let val = ctx.get_raw(0);
            match val {
                rusqlite::types::ValueRef::Null => Ok(None::<f64>),
                rusqlite::types::ValueRef::Integer(i) => Ok(Some((i as f64).exp())),
                rusqlite::types::ValueRef::Real(f) => Ok(Some(f.exp())),
                _ => Ok(None),
            }
        },
    )?;
    conn.create_scalar_function(
        "pow",
        2,
        FunctionFlags::SQLITE_UTF8 | FunctionFlags::SQLITE_DETERMINISTIC,
        |ctx| {
            let base = match ctx.get_raw(0) {
                rusqlite::types::ValueRef::Null => return Ok(None::<f64>),
                rusqlite::types::ValueRef::Integer(i) => i as f64,
                rusqlite::types::ValueRef::Real(f) => f,
                _ => return Ok(None),
            };
            let exp = match ctx.get_raw(1) {
                rusqlite::types::ValueRef::Null => return Ok(None),
                rusqlite::types::ValueRef::Integer(i) => i as f64,
                rusqlite::types::ValueRef::Real(f) => f,
                _ => return Ok(None),
            };
            Ok(Some(base.powf(exp)))
        },
    )?;
    Ok(())
}

/// Path of the sidecar writer lockfile for a given db path.
fn writer_lock_path(db_path: &str) -> PathBuf {
    PathBuf::from(format!("{db_path}.writer-lock"))
}

/// Attempt to acquire an exclusive advisory flock on the writer-lock
/// sidecar file. Returns the locked ``File`` handle on success.
///
/// Fails with a descriptive error if another process already holds the
/// lock. The error message includes the holder's PID (read from the
/// lockfile contents) so operators know which process to stop.
///
/// The flock is held for as long as the returned ``File`` stays alive.
/// Kernel releases it on process exit even if the process crashes or is
/// SIGKILL'd, so the lockfile never becomes stale in practice.
fn acquire_writer_lock(db_path: &str) -> PyResult<File> {
    let lock_path = writer_lock_path(db_path);
    let mut file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)
        .map_err(|e| {
            PyRuntimeError::new_err(format!(
                "aria-db: cannot open writer lockfile {}: {e}",
                lock_path.display()
            ))
        })?;

    match file.try_lock_exclusive() {
        Ok(()) => {
            // Record our PID so a human running `lsof`/`cat` can see who
            // holds the DB. Overwrite (not append) — one PID per file.
            let pid = std::process::id();
            let _ = file.seek(SeekFrom::Start(0));
            let _ = file.set_len(0);
            let _ = writeln!(&file, "{pid}");
            let _ = file.flush();
            Ok(file)
        }
        Err(_) => {
            // Read the existing holder PID for the error message.
            let mut holder = String::new();
            let _ = file.seek(SeekFrom::Start(0));
            let _ = file.read_to_string(&mut holder);
            let holder = holder.trim();
            let holder_hint = if holder.is_empty() {
                "<unknown>".to_string()
            } else {
                holder.to_string()
            };
            Err(PyRuntimeError::new_err(format!(
                "aria-db: another process already holds the writer lock on {} \
                 (lock held by PID {holder_hint}, lockfile {}). Running two aria-db \
                 writers against the same database causes orphaned WAL files and \
                 silent data loss — stop the other process first, or use the \
                 read-only manager (get_manager_readonly) if you only need to query.",
                db_path,
                lock_path.display(),
            )))
        }
    }
}

fn open_connection(db_path: &str) -> Result<Connection, PyErr> {
    let flags = OpenFlags::SQLITE_OPEN_READ_WRITE
        | OpenFlags::SQLITE_OPEN_CREATE
        | OpenFlags::SQLITE_OPEN_NO_MUTEX;

    let conn = Connection::open_with_flags(db_path, flags)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to open database: {e}")))?;

    // DELETE journal mode — eliminates the entire WAL/SHM file class.
    //
    // The original module used WAL mode (Write-Ahead Logging) because it
    // theoretically allows concurrent readers during writes.  But aria-db
    // serializes ALL access through a single Rust mutex, so there is
    // never concurrent reader+writer activity — WAL's only benefit is
    // gone.  Meanwhile, WAL mode creates -wal and -shm sidecar files
    // whose lifecycle is fatally fragile:
    //
    //   * Any second connection (even read-only) can trigger SHM teardown
    //     on close, unlinking the -wal/-shm files while the server holds
    //     open file descriptors to the now-deleted inodes.
    //   * The server continues writing to the deleted WAL via its stale
    //     FD — all subsequent writes are invisible to every reader.
    //   * This caused 16+ hours of silent program_result data loss on
    //     2026-04-16, and recurred despite flock hardening and immutable
    //     read-only mode because SQLite's internal SHM refcounting has
    //     race conditions that external advisory locks cannot prevent.
    //
    // DELETE journal mode uses a single rollback journal file that is
    // created per-transaction and deleted on commit.  There is no SHM,
    // no persistent WAL, and no teardown race.  The single-connection
    // architecture means the performance difference is negligible.
    //
    // foreign_keys is deliberately OFF — see test_native_conn_fk_default.py.
    conn.execute_batch(
        "PRAGMA journal_mode=DELETE;
         PRAGMA synchronous=NORMAL;
         PRAGMA foreign_keys=OFF;
         PRAGMA busy_timeout=15000;",
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Failed to set pragmas: {e}")))?;

    // Register math functions (SQRT, LOG, EXP, POW) that Python's sqlite3 provides.
    register_math_functions(&conn)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to register math functions: {e}")))?;

    Ok(conn)
}

/// Open a read-only connection. Used by tests, backfill scripts, and
/// admin tools that only query.
///
/// **Key hardening**: this connection opens in ``immutable=1`` mode via
/// a URI filename.  Immutable connections never access the WAL or SHM
/// files at all — they read directly from the main `.db` file as a
/// frozen snapshot.  This is the only way to guarantee that a short-
/// lived reader process cannot trigger SHM teardown (which unlinks the
/// WAL and strands the long-running writer's in-flight data).
///
/// The trade-off: the read-only connection only sees data that has been
/// checkpointed from the WAL into the main database file. Un-checkpointed
/// rows in the WAL are invisible. For monitoring/reporting this is
/// acceptable — the data appears after the next checkpoint. For test
/// assertions that need to see freshly-written rows, use a temp DB path
/// instead of the production database.
fn open_connection_readonly(db_path: &str) -> Result<Connection, PyErr> {
    // Use a URI filename with immutable=1 to completely bypass WAL/SHM.
    // This requires SQLITE_OPEN_URI in the flags.
    let uri = format!("file:{}?immutable=1", db_path);
    let flags = OpenFlags::SQLITE_OPEN_READ_ONLY
        | OpenFlags::SQLITE_OPEN_NO_MUTEX
        | OpenFlags::SQLITE_OPEN_URI;
    let conn = Connection::open_with_flags(&uri, flags).map_err(|e| {
        PyRuntimeError::new_err(format!("Failed to open database read-only: {e}"))
    })?;
    conn.execute_batch(
        "PRAGMA query_only=ON;
         PRAGMA busy_timeout=15000;",
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Failed to set pragmas: {e}")))?;
    register_math_functions(&conn).map_err(|e| {
        PyRuntimeError::new_err(format!("Failed to register math functions: {e}"))
    })?;
    Ok(conn)
}

#[pymethods]
impl ConnectionManager {
    #[new]
    fn new(db_path: &str) -> PyResult<Self> {
        // Acquire the writer flock FIRST. If another process holds it, we
        // fail here with a clear error rather than proceeding to open
        // SQLite and clobber the WAL.
        let lock = acquire_writer_lock(db_path)?;

        let conn = match open_connection(db_path) {
            Ok(c) => c,
            Err(e) => {
                // Drop the lock by dropping the file handle so we don't
                // leave a zombie writer-lock on a failed init.
                drop(lock);
                return Err(e);
            }
        };

        Ok(Self {
            inner: Arc::new(Mutex::new(DbInner { conn })),
            writer_tx: Mutex::new(None),
            writer_started: AtomicBool::new(false),
            db_path: db_path.to_string(),
            read_only: false,
            writer_lock: Mutex::new(Some(lock)),
        })
    }

    /// Factory for a read-only manager. Callers get full query access but
    /// any submit_write / submit_write_many will raise. Use this for tests
    /// and for tools that should never accidentally touch the WAL.
    #[staticmethod]
    fn new_readonly(db_path: &str) -> PyResult<Self> {
        let conn = open_connection_readonly(db_path)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(DbInner { conn })),
            writer_tx: Mutex::new(None),
            writer_started: AtomicBool::new(false),
            db_path: db_path.to_string(),
            read_only: true,
            writer_lock: Mutex::new(None),
        })
    }

    /// Whether this manager is in read-only mode. Exposed so Python can
    /// assert its opened the expected kind of connection.
    #[getter]
    fn read_only(&self) -> bool {
        self.read_only
    }

    /// Whether this manager currently holds the writer flock. True for
    /// read-write managers, false for read-only. Exposed for Python-side
    /// health checks and regression tests.
    #[getter]
    fn holds_writer_lock(&self) -> bool {
        self.writer_lock.lock().is_some()
    }

    /// Execute a SQL statement (no result rows).
    /// Used for INSERT, UPDATE, DELETE, CREATE, etc.
    fn execute(&self, sql: &str, params: &Bound<'_, pyo3::PyAny>) -> PyResult<usize> {
        let values = py_params(params)?;
        let inner = self.inner.lock();
        inner
            .conn
            .execute(sql, params_from_iter(values.iter()))
            .map_err(|e| PyRuntimeError::new_err(format!("execute error: {e}")))
    }

    /// Execute a SQL statement with multiple parameter sets.
    fn executemany(
        &self,
        sql: &str,
        params_list: &Bound<'_, pyo3::PyAny>,
    ) -> PyResult<usize> {
        let outer: Vec<Bound<'_, pyo3::PyAny>> = params_list.extract()?;
        let param_sets: Vec<Vec<Value>> = outer
            .iter()
            .map(py_params)
            .collect::<Result<Vec<_>, _>>()?;
        let inner = self.inner.lock();
        let mut total = 0usize;
        for values in &param_sets {
            total += inner
                .conn
                .execute(sql, params_from_iter(values.iter()))
                .map_err(|e| PyRuntimeError::new_err(format!("executemany error: {e}")))?;
        }
        Ok(total)
    }

    /// Execute a SQL script (multiple statements separated by semicolons).
    fn executescript(&self, sql: &str) -> PyResult<()> {
        let inner = self.inner.lock();
        inner
            .conn
            .execute_batch(sql)
            .map_err(|e| PyRuntimeError::new_err(format!("executescript error: {e}")))
    }

    /// Execute a SELECT and return all rows as list of dicts.
    fn fetchall(
        &self,
        py: Python<'_>,
        sql: &str,
        params: &Bound<'_, pyo3::PyAny>,
    ) -> PyResult<PyObject> {
        let values = py_params(params)?;
        let inner = self.inner.lock();
        let mut stmt = inner
            .conn
            .prepare(sql)
            .map_err(|e| PyRuntimeError::new_err(format!("prepare error: {e}")))?;
        let col_names: Vec<String> = stmt.column_names().iter().map(|s| s.to_string()).collect();
        let rows = stmt
            .query_map(params_from_iter(values.iter()), |row| {
                let mut cols = Vec::with_capacity(col_names.len());
                for i in 0..col_names.len() {
                    cols.push(rusqlite_value_to_ours(&row.get_ref(i)?));
                }
                Ok(cols)
            })
            .map_err(|e| PyRuntimeError::new_err(format!("query error: {e}")))?;

        let result = PyList::empty_bound(py);
        for row_result in rows {
            let cols = row_result
                .map_err(|e| PyRuntimeError::new_err(format!("row error: {e}")))?;
            let dict = PyDict::new_bound(py);
            for (name, val) in col_names.iter().zip(cols.iter()) {
                dict.set_item(name, val.to_python(py))?;
            }
            result.append(dict)?;
        }
        Ok(result.into())
    }

    /// Execute a SELECT and return the first row as a dict, or None.
    fn fetchone(
        &self,
        py: Python<'_>,
        sql: &str,
        params: &Bound<'_, pyo3::PyAny>,
    ) -> PyResult<PyObject> {
        let values = py_params(params)?;
        let inner = self.inner.lock();
        let mut stmt = inner
            .conn
            .prepare(sql)
            .map_err(|e| PyRuntimeError::new_err(format!("prepare error: {e}")))?;
        let col_names: Vec<String> = stmt.column_names().iter().map(|s| s.to_string()).collect();
        let mut rows = stmt
            .query_map(params_from_iter(values.iter()), |row| {
                let mut cols = Vec::with_capacity(col_names.len());
                for i in 0..col_names.len() {
                    cols.push(rusqlite_value_to_ours(&row.get_ref(i)?));
                }
                Ok(cols)
            })
            .map_err(|e| PyRuntimeError::new_err(format!("query error: {e}")))?;

        match rows.next() {
            Some(Ok(cols)) => {
                let dict = PyDict::new_bound(py);
                for (name, val) in col_names.iter().zip(cols.iter()) {
                    dict.set_item(name, val.to_python(py))?;
                }
                Ok(dict.into())
            }
            Some(Err(e)) => Err(PyRuntimeError::new_err(format!("row error: {e}"))),
            None => Ok(py.None()),
        }
    }

    /// Execute a SELECT and return all rows as list of tuples.
    /// This matches sqlite3.Row behavior when accessed by index.
    fn fetchall_tuples(
        &self,
        py: Python<'_>,
        sql: &str,
        params: &Bound<'_, pyo3::PyAny>,
    ) -> PyResult<PyObject> {
        let values = py_params(params)?;
        let inner = self.inner.lock();
        let mut stmt = inner
            .conn
            .prepare(sql)
            .map_err(|e| PyRuntimeError::new_err(format!("prepare error: {e}")))?;
        let col_count = stmt.column_count();
        let rows = stmt
            .query_map(params_from_iter(values.iter()), |row| {
                let mut cols = Vec::with_capacity(col_count);
                for i in 0..col_count {
                    cols.push(rusqlite_value_to_ours(&row.get_ref(i)?));
                }
                Ok(cols)
            })
            .map_err(|e| PyRuntimeError::new_err(format!("query error: {e}")))?;

        let result = PyList::empty_bound(py);
        for row_result in rows {
            let cols = row_result
                .map_err(|e| PyRuntimeError::new_err(format!("row error: {e}")))?;
            let py_vals: Vec<PyObject> = cols.iter().map(|v| v.to_python(py)).collect();
            result.append(PyTuple::new_bound(py, py_vals))?;
        }
        Ok(result.into())
    }

    /// Execute a SELECT and return the first row as a tuple, or None.
    fn fetchone_tuple(
        &self,
        py: Python<'_>,
        sql: &str,
        params: &Bound<'_, pyo3::PyAny>,
    ) -> PyResult<PyObject> {
        let values = py_params(params)?;
        let inner = self.inner.lock();
        let mut stmt = inner
            .conn
            .prepare(sql)
            .map_err(|e| PyRuntimeError::new_err(format!("prepare error: {e}")))?;
        let col_count = stmt.column_count();
        let mut rows = stmt
            .query_map(params_from_iter(values.iter()), |row| {
                let mut cols = Vec::with_capacity(col_count);
                for i in 0..col_count {
                    cols.push(rusqlite_value_to_ours(&row.get_ref(i)?));
                }
                Ok(cols)
            })
            .map_err(|e| PyRuntimeError::new_err(format!("query error: {e}")))?;

        match rows.next() {
            Some(Ok(cols)) => {
                let py_vals: Vec<PyObject> = cols.iter().map(|v| v.to_python(py)).collect();
                Ok(PyTuple::new_bound(py, py_vals).into())
            }
            Some(Err(e)) => Err(PyRuntimeError::new_err(format!("row error: {e}"))),
            None => Ok(py.None()),
        }
    }

    /// Commit the current transaction.
    fn commit(&self) -> PyResult<()> {
        // In autocommit (default for rusqlite), this is a no-op, but we
        // expose it for API compatibility with the existing notebook code.
        // If we need explicit transactions, we handle them internally.
        Ok(())
    }

    /// Submit a write to the async writer thread (non-blocking).
    /// Matches LabNotebook._submit_write semantics.
    fn submit_write(&self, sql: &str, params: &Bound<'_, pyo3::PyAny>) -> PyResult<()> {
        if self.read_only {
            return Err(PyRuntimeError::new_err(
                "aria-db: submit_write called on a read-only manager. Open the \
                 manager via get_manager() (not get_manager_readonly()) if you \
                 need write access.",
            ));
        }
        self.ensure_writer()?;
        let values = py_params(params)?;

        // Check if this is an executemany-style call (list of lists/tuples)
        let msg = if values.len() > 0 {
            // Check if first param is itself a sequence (executemany pattern)
            // The Python side passes list-of-tuples for executemany
            WriteMsg::Execute {
                sql: sql.to_string(),
                params: values,
            }
        } else {
            WriteMsg::Execute {
                sql: sql.to_string(),
                params: values,
            }
        };

        let guard = self.writer_tx.lock();
        if let Some(tx) = guard.as_ref() {
            tx.send(msg).map_err(|e| {
                PyRuntimeError::new_err(format!("Writer channel closed: {e}"))
            })?;
        }
        Ok(())
    }

    /// Submit an executemany to the async writer thread (non-blocking).
    fn submit_write_many(
        &self,
        sql: &str,
        params_list: &Bound<'_, pyo3::PyAny>,
    ) -> PyResult<()> {
        if self.read_only {
            return Err(PyRuntimeError::new_err(
                "aria-db: submit_write_many called on a read-only manager.",
            ));
        }
        self.ensure_writer()?;
        let outer: Vec<Bound<'_, pyo3::PyAny>> = params_list.extract()?;
        let param_sets: Vec<Vec<Value>> = outer
            .iter()
            .map(py_params)
            .collect::<Result<Vec<_>, _>>()?;

        let guard = self.writer_tx.lock();
        if let Some(tx) = guard.as_ref() {
            tx.send(WriteMsg::ExecuteMany {
                sql: sql.to_string(),
                params_list: param_sets,
            })
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Writer channel closed: {e}"))
            })?;
        }
        Ok(())
    }

    /// Block until the async write queue is drained and committed.
    /// Matches LabNotebook.flush_writes semantics.
    #[pyo3(signature = (timeout_secs=None))]
    fn flush_writes(&self, timeout_secs: Option<f64>) -> PyResult<()> {
        let timeout = Duration::from_secs_f64(timeout_secs.unwrap_or(5.0));
        if !self.writer_started.load(Ordering::Relaxed) {
            return Ok(());  // No writer thread, nothing to flush
        }
        let done = Arc::new((parking_lot::Mutex::new(false), parking_lot::Condvar::new()));
        {
            let guard = self.writer_tx.lock();
            if let Some(tx) = guard.as_ref() {
                tx.send(WriteMsg::Flush { done: done.clone() })
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!("Writer channel closed: {e}"))
                    })?;
            } else {
                return Ok(());
            }
        }
        // Wait for the writer thread to signal completion.
        let (lock, cvar) = &*done;
        let mut finished = lock.lock();
        if !*finished {
            cvar.wait_for(&mut finished, timeout);
        }
        if !*finished {
            return Err(PyRuntimeError::new_err("flush_writes timed out"));
        }
        Ok(())
    }

    /// Get the database path.
    #[getter]
    fn db_path(&self) -> &str {
        &self.db_path
    }

    /// Check if the connection is alive.
    fn ping(&self) -> PyResult<bool> {
        let inner = self.inner.lock();
        match inner.conn.execute_batch("SELECT 1") {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Run PRAGMA table_info for a table. Returns list of (cid, name, type, notnull, dflt, pk).
    fn table_info(
        &self,
        py: Python<'_>,
        table: &str,
    ) -> PyResult<PyObject> {
        // Validate table name to prevent injection (only alnum and underscore)
        if !table.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Err(PyValueError::new_err("Invalid table name"));
        }
        let sql = format!("PRAGMA table_info({})", table);
        self.fetchall_tuples(py, &sql, &py.None().into_bound(py))
    }

    /// Checkpoint the WAL. Call this periodically or on clean shutdown.
    fn checkpoint(&self) -> PyResult<()> {
        let inner = self.inner.lock();
        inner
            .conn
            .execute_batch("PRAGMA wal_checkpoint(PASSIVE)")
            .map_err(|e| PyRuntimeError::new_err(format!("checkpoint error: {e}")))
    }

    /// Stop the writer thread. Does NOT close the connection.
    fn stop_writer(&self) -> PyResult<()> {
        if !self.writer_started.load(Ordering::Relaxed) {
            return Ok(());
        }
        let guard = self.writer_tx.lock();
        if let Some(tx) = guard.as_ref() {
            let _ = tx.send(WriteMsg::Stop);
        }
        self.writer_started.store(false, Ordering::Relaxed);
        Ok(())
    }
}

impl ConnectionManager {
    fn ensure_writer(&self) -> PyResult<()> {
        if self.writer_started.load(Ordering::Relaxed) {
            return Ok(());
        }

        let mut guard = self.writer_tx.lock();
        // Double-check after acquiring lock.
        if self.writer_started.load(Ordering::Relaxed) {
            return Ok(());
        }

        let (tx, rx) = bounded::<WriteMsg>(4096);
        let inner = self.inner.clone();

        thread::Builder::new()
            .name("aria-db-writer".into())
            .spawn(move || {
                // Use the SAME connection as the reader via the shared mutex.
                // This guarantees exactly ONE connection to the DB exists,
                // which is the whole point — preventing SHM teardown from
                // a second connection's close/open lifecycle.
                for msg in rx {
                    match msg {
                        WriteMsg::Execute { sql, params } => {
                            let guard = inner.lock();
                            if let Err(e) =
                                guard.conn.execute(&sql, params_from_iter(params.iter()))
                            {
                                // stderr + level marker so log greps find it.
                                // Include a slightly longer SQL prefix (200 chars)
                                // because 80 is not enough to identify which
                                // INSERT failed.
                                eprintln!(
                                    "[aria-db writer] ERROR execute failed: {e} — sql[0..{}]: {}",
                                    sql.len().min(200),
                                    &sql[..sql.len().min(200)],
                                );
                            }
                        }
                        WriteMsg::ExecuteMany { sql, params_list } => {
                            let guard = inner.lock();
                            for params in &params_list {
                                if let Err(e) =
                                    guard.conn.execute(&sql, params_from_iter(params.iter()))
                                {
                                    eprintln!(
                                        "[aria-db writer] ERROR executemany failed: {e} — sql[0..{}]: {}",
                                        sql.len().min(200),
                                        &sql[..sql.len().min(200)],
                                    );
                                    break;
                                }
                            }
                        }
                        WriteMsg::Flush { done } => {
                            // Acquire and release the lock to ensure all
                            // prior writes are visible, then signal caller.
                            let _guard = inner.lock();
                            let (lock, cvar) = &*done;
                            let mut finished = lock.lock();
                            *finished = true;
                            cvar.notify_all();
                        }
                        WriteMsg::Stop => break,
                    }
                }
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to start writer thread: {e}")))?;

        *guard = Some(tx);
        self.writer_started.store(true, Ordering::Relaxed);
        Ok(())
    }
}

impl Drop for ConnectionManager {
    fn drop(&mut self) {
        // Stop the writer thread on drop.
        if self.writer_started.load(Ordering::Relaxed) {
            let guard = self.writer_tx.lock();
            if let Some(tx) = guard.as_ref() {
                let _ = tx.send(WriteMsg::Stop);
            }
        }
        // Intentionally do NOT close the connection.
        // The connection lives as long as the process to prevent SHM teardown.
        // When the process exits, the OS reclaims all resources.
    }
}

/// Global singleton manager — ensures only one ConnectionManager per db path.
static MANAGERS: std::sync::LazyLock<Mutex<HashMap<String, Py<ConnectionManager>>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

/// Read-only managers are keyed separately so one process can hold both a
/// read-write manager on its own path AND a read-only manager on another
/// (or the same) path. In practice the dashboard only uses read-write.
static READONLY_MANAGERS: std::sync::LazyLock<
    Mutex<HashMap<String, Py<ConnectionManager>>>,
> = std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

/// Get or create the singleton **read-write** ConnectionManager for a
/// database path. Takes the writer flock; fails fast if another process
/// already holds it.
#[pyfunction]
fn get_manager(py: Python<'_>, db_path: &str) -> PyResult<Py<ConnectionManager>> {
    let canonical = std::fs::canonicalize(db_path)
        .unwrap_or_else(|_| std::path::PathBuf::from(db_path));
    let key = canonical.to_string_lossy().to_string();

    let mut managers = MANAGERS.lock();
    if let Some(existing) = managers.get(&key) {
        return Ok(existing.clone_ref(py));
    }

    let mgr = Py::new(py, ConnectionManager::new(db_path)?)?;
    managers.insert(key, mgr.clone_ref(py));
    Ok(mgr)
}

/// Get or create a **read-only** ConnectionManager. Use this from tests,
/// backfill/audit scripts, and admin tools — it cannot trigger the
/// close-time WAL teardown that previously stranded writer data, and
/// write attempts raise instead of silently failing.
#[pyfunction]
fn get_manager_readonly(
    py: Python<'_>,
    db_path: &str,
) -> PyResult<Py<ConnectionManager>> {
    let canonical = std::fs::canonicalize(db_path)
        .unwrap_or_else(|_| std::path::PathBuf::from(db_path));
    let key = canonical.to_string_lossy().to_string();

    let mut managers = READONLY_MANAGERS.lock();
    if let Some(existing) = managers.get(&key) {
        return Ok(existing.clone_ref(py));
    }

    let mgr = Py::new(py, ConnectionManager::new_readonly(db_path)?)?;
    managers.insert(key, mgr.clone_ref(py));
    Ok(mgr)
}

/// Python module definition.
#[pymodule]
fn aria_db(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ConnectionManager>()?;
    m.add_function(wrap_pyfunction!(get_manager, m)?)?;
    m.add_function(wrap_pyfunction!(get_manager_readonly, m)?)?;
    Ok(())
}
