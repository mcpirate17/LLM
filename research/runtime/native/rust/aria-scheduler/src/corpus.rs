use std::collections::HashMap;
use std::collections::HashSet;
use std::fs;
use std::path::Path;

use rusqlite::{Connection, OptionalExtension};
use serde::Serialize;
use serde_json::Value;

use crate::error::AriaError;
use crate::notebook_graph::NotebookGraph;

#[derive(Clone)]
struct GraphTrainingInput {
    graph_json: String,
    stage1_passed: bool,
    wikitext_perplexity: Option<f64>,
    wikitext_metric_version: String,
    loss_ratio: Option<f64>,
    stage0_passed: bool,
    stage05_passed: bool,
    timestamp: f64,
}

#[derive(Clone)]
struct PredictorTrainingInput {
    graph_json: String,
    fingerprint_json: String,
    novelty_score: Option<f64>,
    structural_novelty: Option<f64>,
    target_loss_ratio: f64,
    tier: String,
    timestamp: f64,
}

#[derive(Serialize)]
pub struct GraphTrainingRow {
    pub canonical_fingerprint: String,
    pub graph_json: String,
    pub stage1_any_passed: bool,
    pub stage1_pass_rate: f64,
    pub stage0_any_passed: bool,
    pub stage05_any_passed: bool,
    pub wikitext_perplexity_best: Option<f64>,
    pub loss_ratio_best: Option<f64>,
    pub n_rows: usize,
    pub latest_timestamp: f64,
}

#[derive(Serialize)]
pub struct PredictorTrainingRow {
    pub canonical_fingerprint: String,
    pub fingerprint_json: String,
    pub novelty_score: Option<f64>,
    pub structural_novelty: Option<f64>,
    pub target_loss_ratio: f64,
    pub tier: String,
    pub n_rows: usize,
}

struct GraphAccumulator {
    canonical_fingerprint: String,
    representative: Option<GraphTrainingInput>,
    n_rows: usize,
    n_stage1_passed: usize,
    stage0_any_passed: bool,
    stage05_any_passed: bool,
    wikitext_perplexity_best: Option<f64>,
    loss_ratio_best: Option<f64>,
    latest_timestamp: f64,
}

impl GraphAccumulator {
    fn new(canonical_fingerprint: String) -> Self {
        Self {
            canonical_fingerprint,
            representative: None,
            n_rows: 0,
            n_stage1_passed: 0,
            stage0_any_passed: false,
            stage05_any_passed: false,
            wikitext_perplexity_best: None,
            loss_ratio_best: None,
            latest_timestamp: 0.0,
        }
    }

    fn absorb(&mut self, row: GraphTrainingInput) {
        self.n_rows += 1;
        if row.stage1_passed {
            self.n_stage1_passed += 1;
        }
        self.stage0_any_passed |= row.stage0_passed;
        self.stage05_any_passed |= row.stage05_passed;
        if row.wikitext_metric_version == "bpe_eval_v1" {
            self.wikitext_perplexity_best =
                min_option(self.wikitext_perplexity_best, row.wikitext_perplexity);
        }
        self.loss_ratio_best = min_option(self.loss_ratio_best, row.loss_ratio);
        self.latest_timestamp = self.latest_timestamp.max(row.timestamp);

        let replace = self
            .representative
            .as_ref()
            .map(|current| graph_row_rank(&row) < graph_row_rank(current))
            .unwrap_or(true);
        if replace {
            self.representative = Some(row);
        }
    }

    fn finish(self) -> Result<GraphTrainingRow, AriaError> {
        let representative = self.representative.ok_or_else(|| {
            AriaError::ExecutionFailed("graph corpus accumulator missing representative".into())
        })?;
        Ok(GraphTrainingRow {
            canonical_fingerprint: self.canonical_fingerprint,
            graph_json: representative.graph_json,
            stage1_any_passed: self.n_stage1_passed > 0,
            stage1_pass_rate: self.n_stage1_passed as f64 / self.n_rows.max(1) as f64,
            stage0_any_passed: self.stage0_any_passed,
            stage05_any_passed: self.stage05_any_passed,
            wikitext_perplexity_best: self.wikitext_perplexity_best,
            loss_ratio_best: self.loss_ratio_best,
            n_rows: self.n_rows,
            latest_timestamp: self.latest_timestamp,
        })
    }
}

struct PredictorAccumulator {
    canonical_fingerprint: String,
    representative: Option<PredictorTrainingInput>,
    n_rows: usize,
}

impl PredictorAccumulator {
    fn new(canonical_fingerprint: String) -> Self {
        Self {
            canonical_fingerprint,
            representative: None,
            n_rows: 0,
        }
    }

    fn absorb(&mut self, row: PredictorTrainingInput) {
        self.n_rows += 1;
        let replace = self
            .representative
            .as_ref()
            .map(|current| predictor_row_rank(&row) < predictor_row_rank(current))
            .unwrap_or(true);
        if replace {
            self.representative = Some(row);
        }
    }

    fn finish(self) -> Result<PredictorTrainingRow, AriaError> {
        let representative = self.representative.ok_or_else(|| {
            AriaError::ExecutionFailed("predictor corpus accumulator missing representative".into())
        })?;
        Ok(PredictorTrainingRow {
            canonical_fingerprint: self.canonical_fingerprint,
            fingerprint_json: representative.fingerprint_json,
            novelty_score: representative.novelty_score,
            structural_novelty: representative.structural_novelty,
            target_loss_ratio: representative.target_loss_ratio,
            tier: representative.tier,
            n_rows: self.n_rows,
        })
    }
}

pub fn fingerprint_notebook_graph_json(graph_json: &str) -> Result<String, AriaError> {
    NotebookGraph::from_json(graph_json)?.fingerprint()
}

pub fn build_graph_training_corpus_json(db_path: &Path) -> Result<String, AriaError> {
    let conn = open_notebook_db(db_path)?;
    let program_results_columns = table_columns(&conn, "program_results")?;
    let metric_version_select =
        if program_results_columns.contains("screening_wikitext_metric_version") {
            ", COALESCE(screening_wikitext_metric_version, '')"
        } else {
            ", ''"
        };
    let mut query = String::from(
        "
            SELECT graph_json, stage1_passed, wikitext_perplexity, loss_ratio,
                   stage0_passed, stage05_passed, timestamp
            FROM program_results
            WHERE TRIM(COALESCE(graph_json, '')) <> ''
              AND graph_json <> '{}'
        ",
    );
    query = query.replace(
        "stage0_passed, stage05_passed, timestamp",
        &format!(
            "stage0_passed, stage05_passed, timestamp{}",
            metric_version_select
        ),
    );
    if has_trust_columns(&program_results_columns) {
        query.push_str(
            "
              AND COALESCE(trust_label, '') IN ('candidate_screening', 'candidate_grade', 'reference')
              AND COALESCE(comparability_label, '') IN ('screening_only', 'candidate_comparable', 'reference_comparable')
            ",
        );
    }
    push_non_byte_training_data_filters(&mut query, &program_results_columns, None);
    let mut stmt = conn
        .prepare(&query)
        .map_err(|e| AriaError::ExecutionFailed(e.to_string()))?;

    let rows = {
        let rows = stmt
            .query_map([], |row| {
                Ok(GraphTrainingInput {
                    graph_json: row.get(0)?,
                    stage1_passed: row.get::<_, Option<i64>>(1)?.unwrap_or(0) != 0,
                    wikitext_perplexity: row.get(2)?,
                    loss_ratio: row.get(3)?,
                    stage0_passed: row.get::<_, Option<i64>>(4)?.unwrap_or(0) != 0,
                    stage05_passed: row.get::<_, Option<i64>>(5)?.unwrap_or(0) != 0,
                    timestamp: row.get::<_, Option<f64>>(6)?.unwrap_or(0.0),
                    wikitext_metric_version: row.get::<_, Option<String>>(7)?.unwrap_or_default(),
                })
            })
            .map_err(|e| AriaError::ExecutionFailed(e.to_string()))?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| AriaError::ExecutionFailed(e.to_string()))?
    };
    drop(stmt);

    let mut groups: HashMap<String, GraphAccumulator> = HashMap::new();
    for mut row in rows {
        row.graph_json = resolve_graph_json_value(&conn, db_path, &row.graph_json)?;
        let canonical_fingerprint = fingerprint_notebook_graph_json(&row.graph_json)?;
        groups
            .entry(canonical_fingerprint.clone())
            .or_insert_with(|| GraphAccumulator::new(canonical_fingerprint))
            .absorb(row);
    }

    let mut deduped: Vec<GraphTrainingRow> = groups
        .into_values()
        .map(GraphAccumulator::finish)
        .collect::<Result<Vec<_>, _>>()?;
    deduped.sort_by(|a, b| a.canonical_fingerprint.cmp(&b.canonical_fingerprint));
    serde_json::to_string(&deduped).map_err(|e| AriaError::ExecutionFailed(e.to_string()))
}

pub fn build_predictor_training_corpus_json(db_path: &Path) -> Result<String, AriaError> {
    let conn = open_notebook_db(db_path)?;
    let program_results_columns = table_columns(&conn, "program_results")?;
    let mut query = String::from(
        "
            SELECT pr.graph_json,
                   pr.fingerprint_json,
                   pr.novelty_score,
                   pr.structural_novelty,
                   COALESCE(l.investigation_loss_ratio, pr.loss_ratio) AS target_loss_ratio,
                   COALESCE(l.tier, 'screening') AS tier,
                   pr.timestamp
            FROM program_results pr
            JOIN leaderboard l ON l.result_id = pr.result_id
            WHERE TRIM(COALESCE(pr.graph_json, '')) <> ''
              AND pr.graph_json <> '{}'
              AND pr.fingerprint_json IS NOT NULL
              AND COALESCE(l.investigation_loss_ratio, pr.loss_ratio) IS NOT NULL
        ",
    );
    if has_trust_columns(&program_results_columns) {
        query.push_str(
            "
              AND COALESCE(pr.trust_label, '') IN ('candidate_grade', 'reference')
              AND COALESCE(pr.comparability_label, '') IN ('candidate_comparable', 'reference_comparable')
            ",
        );
    }
    push_non_byte_training_data_filters(&mut query, &program_results_columns, Some("pr"));
    let mut stmt = conn
        .prepare(&query)
        .map_err(|e| AriaError::ExecutionFailed(e.to_string()))?;

    let rows = {
        let rows = stmt
            .query_map([], |row| {
                Ok(PredictorTrainingInput {
                    graph_json: row.get(0)?,
                    fingerprint_json: row.get(1)?,
                    novelty_score: row.get(2)?,
                    structural_novelty: row.get(3)?,
                    target_loss_ratio: row.get(4)?,
                    tier: row
                        .get::<_, Option<String>>(5)?
                        .unwrap_or_else(|| "screening".into()),
                    timestamp: row.get::<_, Option<f64>>(6)?.unwrap_or(0.0),
                })
            })
            .map_err(|e| AriaError::ExecutionFailed(e.to_string()))?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| AriaError::ExecutionFailed(e.to_string()))?
    };
    drop(stmt);

    let mut groups: HashMap<String, PredictorAccumulator> = HashMap::new();
    for mut row in rows {
        row.graph_json = resolve_graph_json_value(&conn, db_path, &row.graph_json)?;
        let canonical_fingerprint = fingerprint_notebook_graph_json(&row.graph_json)?;
        groups
            .entry(canonical_fingerprint.clone())
            .or_insert_with(|| PredictorAccumulator::new(canonical_fingerprint))
            .absorb(row);
    }

    let mut deduped: Vec<PredictorTrainingRow> = groups
        .into_values()
        .map(PredictorAccumulator::finish)
        .collect::<Result<Vec<_>, _>>()?;
    deduped.sort_by(|a, b| a.canonical_fingerprint.cmp(&b.canonical_fingerprint));
    serde_json::to_string(&deduped).map_err(|e| AriaError::ExecutionFailed(e.to_string()))
}

fn open_notebook_db(db_path: &Path) -> Result<Connection, AriaError> {
    let conn = Connection::open(db_path).map_err(|e| AriaError::ExecutionFailed(e.to_string()))?;
    conn.pragma_update(None, "busy_timeout", 10000)
        .map_err(|e| AriaError::ExecutionFailed(e.to_string()))?;
    let _: Option<String> = conn
        .query_row(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='program_results'",
            [],
            |row| row.get(0),
        )
        .optional()
        .map_err(|e| AriaError::ExecutionFailed(e.to_string()))?;
    Ok(conn)
}

#[derive(Clone)]
struct NotebookArtifactMetadata {
    path: String,
    compression: String,
}

fn resolve_graph_json_value(
    conn: &Connection,
    db_path: &Path,
    raw: &str,
) -> Result<String, AriaError> {
    let Some((artifact_id, pointer_path, pointer_compression)) = parse_artifact_pointer(raw) else {
        return Ok(raw.to_string());
    };

    let metadata = artifact_metadata(conn, &artifact_id)?
        .or_else(|| {
            pointer_path.map(|path| NotebookArtifactMetadata {
                path,
                compression: pointer_compression.unwrap_or_else(|| "zstd".to_string()),
            })
        })
        .ok_or_else(|| {
            AriaError::InvalidIR(format!(
                "graph artifact metadata not found: {}",
                artifact_id
            ))
        })?;

    let artifact_path = notebook_artifact_root(db_path).join(&metadata.path);
    let bytes = fs::read(&artifact_path).map_err(|e| {
        AriaError::ExecutionFailed(format!(
            "failed to read graph artifact {}: {}",
            artifact_path.display(),
            e
        ))
    })?;
    let raw_bytes = match metadata.compression.as_str() {
        "zstd" | "" => zstd::stream::decode_all(&bytes[..]).map_err(|e| {
            AriaError::ExecutionFailed(format!(
                "failed to decompress graph artifact {}: {}",
                artifact_path.display(),
                e
            ))
        })?,
        other => {
            return Err(AriaError::InvalidIR(format!(
                "unsupported graph artifact compression: {}",
                other
            )))
        }
    };
    String::from_utf8(raw_bytes)
        .map_err(|e| AriaError::InvalidIR(format!("graph artifact is not utf-8: {}", e)))
}

fn parse_artifact_pointer(raw: &str) -> Option<(String, Option<String>, Option<String>)> {
    let value: Value = serde_json::from_str(raw).ok()?;
    let object = value.as_object()?;
    let artifact_value = object.get("_notebook_artifact")?;
    let artifact_id = json_value_to_string(artifact_value);
    if artifact_id.trim().is_empty() {
        return None;
    }
    let path = object
        .get("path")
        .map(json_value_to_string)
        .filter(|path| !path.trim().is_empty());
    let compression = object
        .get("compression")
        .map(json_value_to_string)
        .filter(|compression| !compression.trim().is_empty());
    Some((artifact_id, path, compression))
}

fn artifact_metadata(
    conn: &Connection,
    artifact_id: &str,
) -> Result<Option<NotebookArtifactMetadata>, AriaError> {
    if artifact_id.is_empty() {
        return Ok(None);
    }
    match conn
        .query_row(
            "SELECT path, compression FROM notebook_artifacts WHERE artifact_id = ?",
            [artifact_id],
            |row| {
                Ok(NotebookArtifactMetadata {
                    path: row.get(0)?,
                    compression: row
                        .get::<_, Option<String>>(1)?
                        .unwrap_or_else(|| "zstd".into()),
                })
            },
        )
        .optional()
    {
        Ok(metadata) => Ok(metadata),
        Err(rusqlite::Error::SqliteFailure(_, _)) => Ok(None),
        Err(e) => Err(AriaError::ExecutionFailed(e.to_string())),
    }
}

fn notebook_artifact_root(db_path: &Path) -> std::path::PathBuf {
    db_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("artifacts")
        .join("notebook")
}

fn json_value_to_string(value: &Value) -> String {
    match value {
        Value::String(raw) => raw.clone(),
        Value::Number(raw) => raw.to_string(),
        Value::Bool(raw) => raw.to_string(),
        Value::Null => String::new(),
        _ => value.to_string(),
    }
}

fn table_columns(conn: &Connection, table: &str) -> Result<HashSet<String>, AriaError> {
    let mut stmt = conn
        .prepare(&format!("PRAGMA table_info({})", table))
        .map_err(|e| AriaError::ExecutionFailed(e.to_string()))?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(|e| AriaError::ExecutionFailed(e.to_string()))?;
    let mut columns = HashSet::new();
    for row in rows {
        columns.insert(row.map_err(|e| AriaError::ExecutionFailed(e.to_string()))?);
    }
    Ok(columns)
}

fn has_trust_columns(columns: &HashSet<String>) -> bool {
    columns.contains("trust_label") && columns.contains("comparability_label")
}

fn qualified_col(name: &str, alias: Option<&str>) -> String {
    match alias {
        Some(alias) => format!("{}.{}", alias, name),
        None => name.to_string(),
    }
}

fn lower_coalesce_expr(expr: &str) -> String {
    format!("LOWER(COALESCE(CAST({} AS TEXT), ''))", expr)
}

fn json_text_expr(json_col: &str, path: &str) -> String {
    format!(
        "CASE WHEN json_valid(COALESCE({}, '{{}}')) THEN json_extract({}, '{}') ELSE NULL END",
        json_col, json_col, path
    )
}

fn not_contains_byte_markers(expr: &str) -> String {
    let lowered = lower_coalesce_expr(expr);
    format!("({} NOT LIKE '%byte%')", lowered)
}

fn push_non_byte_training_data_filters(
    query: &mut String,
    columns: &HashSet<String>,
    alias: Option<&str>,
) {
    if columns.contains("tokenizer_mode") {
        query.push_str(&format!(
            "\n              AND {} NOT IN ('byte', 'bytes', 'raw_byte', 'raw_bytes')",
            lower_coalesce_expr(&qualified_col("tokenizer_mode", alias))
        ));
    }
    if columns.contains("screening_wikitext_metric_version") {
        query.push_str(&format!(
            "\n              AND {}",
            not_contains_byte_markers(&qualified_col("screening_wikitext_metric_version", alias))
        ));
    }
    if columns.contains("data_provenance_json") {
        let json_col = qualified_col("data_provenance_json", alias);
        for path in [
            "$.tokenizer_mode",
            "$.tokenizer_id",
            "$.tokenizer_version",
            "$.screening_wikitext_metric_version",
            "$.metric_version",
            "$.wikitext_metric_version",
        ] {
            query.push_str(&format!(
                "\n              AND {}",
                not_contains_byte_markers(&json_text_expr(&json_col, path))
            ));
        }
    }
}

fn min_option(current: Option<f64>, candidate: Option<f64>) -> Option<f64> {
    match (current, candidate) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (None, Some(b)) => Some(b),
        (Some(a), None) => Some(a),
        (None, None) => None,
    }
}

fn graph_row_rank(row: &GraphTrainingInput) -> (i32, bool, String, u64) {
    (
        if row.stage1_passed { 0 } else { 1 },
        row.loss_ratio.is_none(),
        sortable_f64_text(row.loss_ratio.unwrap_or(f64::INFINITY)),
        row.timestamp.to_bits(),
    )
}

fn predictor_row_rank(row: &PredictorTrainingInput) -> (i32, String, u64) {
    (
        tier_rank(&row.tier),
        sortable_f64_text(row.target_loss_ratio),
        row.timestamp.to_bits(),
    )
}

fn tier_rank(tier: &str) -> i32 {
    match tier {
        "breakthrough" => 0,
        "validation" => 1,
        "investigation" => 2,
        "investigation_failed" => 3,
        "investigation_fingerprint_incomplete" => 4,
        "screening" => 5,
        "screened_out" => 6,
        _ => 7,
    }
}

fn sortable_f64_text(value: f64) -> String {
    format!("{:020.10}", value)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::thread;
    use std::time::{SystemTime, UNIX_EPOCH};

    use rusqlite::{params, Connection};

    use super::{
        build_graph_training_corpus_json, build_predictor_training_corpus_json,
        fingerprint_notebook_graph_json,
    };

    fn temp_db_path(name: &str) -> std::path::PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{}_{}.sqlite3", name, suffix))
    }

    fn sample_graph_json(metadata: &str) -> String {
        format!(
            r#"{{
                "model_dim":256,
                "nodes":{{
                    "0":{{"id":0,"op_name":"input","input_ids":[],"config":{{}}}},
                    "1":{{"id":1,"op_name":"layernorm","input_ids":[0],"config":{{}}}},
                    "2":{{"id":2,"op_name":"add","input_ids":[0,1],"config":{{}}}}
                }},
                "metadata":{}
            }}"#,
            metadata
        )
    }

    fn write_graph_artifact(
        conn: &Connection,
        db_path: &std::path::Path,
        artifact_id: &str,
        rel_path: &str,
        payload: &str,
    ) -> String {
        let artifact_path = db_path
            .parent()
            .unwrap()
            .join("artifacts")
            .join("notebook")
            .join(rel_path);
        fs::create_dir_all(artifact_path.parent().unwrap()).unwrap();
        let compressed = zstd::stream::encode_all(payload.as_bytes(), 10).unwrap();
        fs::write(&artifact_path, compressed).unwrap();
        conn.execute_batch(
            "
                CREATE TABLE IF NOT EXISTS notebook_artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    table_name TEXT NOT NULL,
                    row_pk TEXT NOT NULL,
                    column_name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    compression TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    sha256_uncompressed TEXT NOT NULL,
                    sha256_compressed TEXT NOT NULL,
                    uncompressed_bytes INTEGER NOT NULL,
                    compressed_bytes INTEGER NOT NULL,
                    created_at REAL NOT NULL
                );
                ",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO notebook_artifacts VALUES (?1, 'program_results', 'r1', 'graph_json', ?2, 'zstd', 'application/json', '', '', 0, 0, 1.0)",
            params![artifact_id, rel_path],
        )
        .unwrap();
        format!(
            r#"{{"_notebook_artifact":"{}","compression":"zstd","path":"{}"}}"#,
            artifact_id, rel_path
        )
    }

    fn run_with_large_stack<F>(name: &str, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let handle = thread::Builder::new()
            .name(name.to_string())
            .stack_size(64 * 1024 * 1024)
            .spawn(f)
            .expect("failed to spawn test thread");
        match handle.join() {
            Ok(()) => {}
            Err(err) => std::panic::resume_unwind(err),
        }
    }

    #[test]
    fn graph_training_corpus_dedupes_metadata_only_repeats() {
        run_with_large_stack(
            "graph_training_corpus_dedupes_metadata_only_repeats",
            || {
                let path = temp_db_path("graph_corpus");
                let conn = Connection::open(&path).unwrap();
                conn.execute_batch(
                    "
                CREATE TABLE program_results (
                    graph_json TEXT,
                    stage1_passed INTEGER,
                    wikitext_perplexity REAL,
                    loss_ratio REAL,
                    stage0_passed INTEGER,
                    stage05_passed INTEGER,
                    timestamp REAL
                );
                ",
                )
                .unwrap();
                conn.execute(
                    "INSERT INTO program_results VALUES (?1, 0, 10.0, 1.2, 1, 0, 1.0)",
                    [sample_graph_json(r#"{"templates_used":["a"]}"#)],
                )
                .unwrap();
                conn.execute(
                    "INSERT INTO program_results VALUES (?1, 1, 8.0, 0.7, 1, 1, 2.0)",
                    [sample_graph_json(
                        r#"{"templates_used":["b"],"lineage":{"parent":"x"}}"#,
                    )],
                )
                .unwrap();

                let payload = build_graph_training_corpus_json(&path).unwrap();
                assert!(payload.contains(r#""n_rows":2"#));
                assert!(payload.contains(r#""stage1_any_passed":true"#));

                let _ = fs::remove_file(path);
            },
        );
    }

    #[test]
    fn graph_training_corpus_resolves_artifact_backed_graph_json() {
        run_with_large_stack(
            "graph_training_corpus_resolves_artifact_backed_graph_json",
            || {
                let path = temp_db_path("graph_corpus_artifact");
                let conn = Connection::open(&path).unwrap();
                conn.execute_batch(
                    "
                CREATE TABLE program_results (
                    graph_json TEXT,
                    stage1_passed INTEGER,
                    wikitext_perplexity REAL,
                    loss_ratio REAL,
                    stage0_passed INTEGER,
                    stage05_passed INTEGER,
                    timestamp REAL
                );
                ",
                )
                .unwrap();
                let graph_json = sample_graph_json(r#"{"templates_used":["artifact"]}"#);
                let pointer = write_graph_artifact(
                    &conn,
                    &path,
                    "artifact_graph",
                    "program_results/artifact_graph/graph_json.json.zst",
                    &graph_json,
                );
                conn.execute(
                    "INSERT INTO program_results VALUES (?1, 1, 8.0, 0.7, 1, 1, 2.0)",
                    [pointer],
                )
                .unwrap();

                let payload = build_graph_training_corpus_json(&path).unwrap();
                let rows: serde_json::Value = serde_json::from_str(&payload).unwrap();
                let rows = rows.as_array().unwrap();
                assert_eq!(rows.len(), 1);
                assert!(rows[0]["graph_json"]
                    .as_str()
                    .unwrap()
                    .contains(r#""nodes""#));
                assert!(!rows[0]["graph_json"]
                    .as_str()
                    .unwrap()
                    .contains("_notebook_artifact"));

                let _ =
                    fs::remove_file(path.parent().unwrap().join(
                        "artifacts/notebook/program_results/artifact_graph/graph_json.json.zst",
                    ));
                let _ = fs::remove_file(path);
            },
        );
    }

    #[test]
    fn predictor_corpus_keeps_best_tier_representative() {
        run_with_large_stack("predictor_corpus_keeps_best_tier_representative", || {
            let path = temp_db_path("predictor_corpus");
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "
                CREATE TABLE program_results (
                    result_id TEXT,
                    graph_json TEXT,
                    fingerprint_json TEXT,
                    novelty_score REAL,
                    structural_novelty REAL,
                    loss_ratio REAL,
                    timestamp REAL
                );
                CREATE TABLE leaderboard (
                    result_id TEXT,
                    investigation_loss_ratio REAL,
                    tier TEXT
                );
                ",
            )
            .unwrap();
            let graph_json = sample_graph_json(r#"{"templates_used":["a"]}"#);
            conn.execute(
                "INSERT INTO program_results VALUES ('r1', ?1, '{\"k\":1}', 1.0, 2.0, 0.9, 1.0)",
                [graph_json.clone()],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO leaderboard VALUES ('r1', NULL, 'screening')",
                [],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO program_results VALUES ('r2', ?1, '{\"k\":2}', 1.5, 2.5, 0.8, 2.0)",
                [graph_json],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO leaderboard VALUES ('r2', 0.4, 'validation')",
                [],
            )
            .unwrap();

            let payload = build_predictor_training_corpus_json(&path).unwrap();
            assert!(payload.contains(r#""tier":"validation""#));
            assert!(payload.contains(r#""target_loss_ratio":0.4"#));

            let _ = fs::remove_file(path);
        });
    }

    #[test]
    fn predictor_corpus_resolves_artifact_backed_graph_json() {
        run_with_large_stack(
            "predictor_corpus_resolves_artifact_backed_graph_json",
            || {
                let path = temp_db_path("predictor_corpus_artifact");
                let conn = Connection::open(&path).unwrap();
                conn.execute_batch(
                    "
                CREATE TABLE program_results (
                    result_id TEXT,
                    graph_json TEXT,
                    fingerprint_json TEXT,
                    novelty_score REAL,
                    structural_novelty REAL,
                    loss_ratio REAL,
                    timestamp REAL
                );
                CREATE TABLE leaderboard (
                    result_id TEXT,
                    investigation_loss_ratio REAL,
                    tier TEXT
                );
                ",
                )
                .unwrap();
                let graph_json = sample_graph_json(r#"{"templates_used":["artifact_predictor"]}"#);
                let pointer = write_graph_artifact(
                    &conn,
                    &path,
                    "predictor_artifact_graph",
                    "program_results/predictor_artifact_graph/graph_json.json.zst",
                    &graph_json,
                );
                conn.execute(
                    "INSERT INTO program_results VALUES ('r1', ?1, '{\"k\":1}', 1.0, 2.0, 0.8, 1.0)",
                    [pointer],
                )
                .unwrap();
                conn.execute(
                    "INSERT INTO leaderboard VALUES ('r1', 0.4, 'validation')",
                    [],
                )
                .unwrap();

                let payload = build_predictor_training_corpus_json(&path).unwrap();
                let rows: serde_json::Value = serde_json::from_str(&payload).unwrap();
                let rows = rows.as_array().unwrap();
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0]["target_loss_ratio"], 0.4);

                let _ = fs::remove_file(
                    path.parent()
                        .unwrap()
                        .join("artifacts/notebook/program_results/predictor_artifact_graph/graph_json.json.zst"),
                );
                let _ = fs::remove_file(path);
            },
        );
    }

    #[test]
    fn corpus_filters_byte_eval_rows() {
        run_with_large_stack("corpus_filters_byte_eval_rows", || {
            let path = temp_db_path("corpus_byte_filter");
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "
                CREATE TABLE program_results (
                    result_id TEXT,
                    graph_json TEXT,
                    fingerprint_json TEXT,
                    novelty_score REAL,
                    structural_novelty REAL,
                    loss_ratio REAL,
                    wikitext_perplexity REAL,
                    stage1_passed INTEGER,
                    stage0_passed INTEGER,
                    stage05_passed INTEGER,
                    timestamp REAL,
                    tokenizer_mode TEXT,
                    screening_wikitext_metric_version TEXT,
                    data_provenance_json TEXT
                );
                CREATE TABLE leaderboard (
                    result_id TEXT,
                    investigation_loss_ratio REAL,
                    tier TEXT
                );
                ",
            )
            .unwrap();

            for (
                result_id,
                graph_json,
                loss_ratio,
                wikitext_perplexity,
                tokenizer_mode,
                metric_version,
                provenance,
            ) in [
                (
                    "good",
                    sample_graph_json(r#"{"templates_used":["good"]}"#),
                    0.8,
                    80.0,
                    "tiktoken",
                    "bpe_eval_v1",
                    r#"{"tokenizer_mode":"tiktoken","screening_wikitext_metric_version":"bpe_eval_v1"}"#,
                ),
                (
                    "byte",
                    sample_graph_json(r#"{"templates_used":["byte"]}"#),
                    0.01,
                    1.0,
                    "byte",
                    "screening_wikitext_v1",
                    r#"{"tokenizer_mode":"byte","screening_wikitext_metric_version":"screening_wikitext_v1"}"#,
                ),
                (
                    "byte_metric",
                    sample_graph_json(r#"{"templates_used":["byte_metric"]}"#),
                    0.02,
                    2.0,
                    "tiktoken",
                    "byte_eval_v1",
                    r#"{"tokenizer_mode":"tiktoken","screening_wikitext_metric_version":"byte_eval_v1"}"#,
                ),
            ] {
                conn.execute(
                    "INSERT INTO program_results VALUES (?1, ?2, '{\"k\":1}', 1.0, 2.0, ?3, ?4, 1, 1, 1, 1.0, ?5, ?6, ?7)",
                    params![
                        result_id,
                        graph_json,
                        loss_ratio,
                        wikitext_perplexity,
                        tokenizer_mode,
                        metric_version,
                        provenance,
                    ],
                )
                .unwrap();
                conn.execute(
                    "INSERT INTO leaderboard VALUES (?1, ?2, 'validation')",
                    params![result_id, loss_ratio],
                )
                .unwrap();
            }

            let graph_payload = build_graph_training_corpus_json(&path).unwrap();
            let graph_rows: serde_json::Value = serde_json::from_str(&graph_payload).unwrap();
            let graph_rows = graph_rows.as_array().unwrap();
            assert_eq!(graph_rows.len(), 1);
            assert_eq!(graph_rows[0]["loss_ratio_best"], 0.8);
            assert_eq!(graph_rows[0]["wikitext_perplexity_best"], 80.0);

            let predictor_payload = build_predictor_training_corpus_json(&path).unwrap();
            let predictor_rows: serde_json::Value =
                serde_json::from_str(&predictor_payload).unwrap();
            let predictor_rows = predictor_rows.as_array().unwrap();
            assert_eq!(predictor_rows.len(), 1);
            assert_eq!(predictor_rows[0]["target_loss_ratio"], 0.8);

            let _ = fs::remove_file(path);
        });
    }

    #[test]
    fn notebook_graph_fingerprint_is_stable_for_reordered_ids() {
        let graph_a = r#"{
            "model_dim":256,
            "nodes":{
                "5":{"id":5,"op_name":"add","input_ids":[2,3],"config":{}},
                "2":{"id":2,"op_name":"input","input_ids":[],"config":{}},
                "3":{"id":3,"op_name":"layernorm","input_ids":[2],"config":{}}
            },
            "metadata":{}
        }"#;
        let graph_b = r#"{
            "model_dim":256,
            "nodes":{
                "10":{"id":10,"op_name":"input","input_ids":[],"config":{}},
                "11":{"id":11,"op_name":"layernorm","input_ids":[10],"config":{}},
                "12":{"id":12,"op_name":"add","input_ids":[10,11],"config":{}}
            },
            "metadata":{}
        }"#;
        let fp_a = fingerprint_notebook_graph_json(graph_a).unwrap();
        let fp_b = fingerprint_notebook_graph_json(graph_b).unwrap();
        assert_eq!(fp_a, fp_b);
    }
}
