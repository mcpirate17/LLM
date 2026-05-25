import React from "react";

// Honest literature provenance for a program, surfaced from the
// `literature_attribution` table + graphs.lit_* columns
// (see research/notes/literature_attribution_2026-05-24.md). match_type tells
// you whether the project reproduced prior work or originated something novel.
const MATCH_COLORS = {
  exact: "var(--accent-green)",
  family: "var(--accent-blue)",
  partial: "var(--accent-amber, #d29922)",
  novel: "var(--accent-purple)",
  unknown: "var(--text-muted)",
};

const MATCH_TITLES = {
  exact: "Faithful reproduction of a published mechanism",
  family: "A variant within this published family",
  partial: "Borrows one key idea, recombined",
  novel: "No published architecture precedent — original to this project",
  unknown: "Structure unavailable (reaped); not classifiable",
};

function MatchBadge({ matchType }) {
  const mt = (matchType || "unknown").toLowerCase();
  const color = MATCH_COLORS[mt] || MATCH_COLORS.unknown;
  return (
    <span
      title={MATCH_TITLES[mt] || ""}
      style={{
        fontSize: 10,
        color,
        border: `1px solid ${color}`,
        borderRadius: 3,
        padding: "0 5px",
        textTransform: "uppercase",
        fontWeight: 600,
        letterSpacing: 0.3,
        whiteSpace: "nowrap",
      }}
    >
      {mt}
    </span>
  );
}

function ModelLink({ name, url }) {
  if (!name) return null;
  if (!url) return <span>{name}</span>;
  return (
    <a
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      style={{ color: "var(--accent-blue)", textDecoration: "none" }}
    >
      {name} ↗
    </a>
  );
}

function Chip({ item }) {
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 5,
        background: "var(--bg-secondary, rgba(255,255,255,0.03))",
        border: "1px solid var(--border)",
        borderRadius: 4,
        padding: "2px 6px",
        fontSize: 11,
        color: "var(--text-secondary)",
      }}
    >
      <span style={{ fontFamily: "monospace" }}>{item.name}</span>
      <span style={{ color: "var(--text-muted)" }}>→</span>
      <ModelLink name={item.external_model_name} url={item.reference_url} />
      <MatchBadge matchType={item.match_type} />
    </span>
  );
}

export default function LiteratureProvenance({ program }) {
  const lit = program && program.literature;
  if (!lit || (!lit.family && !(lit.ops || []).length && !(lit.templates || []).length)) {
    return null;
  }
  const fam = lit.family;
  const sortByMatch = (a, b) => {
    const order = { novel: 0, exact: 1, partial: 2, family: 3, unknown: 4 };
    return (order[a.match_type] ?? 9) - (order[b.match_type] ?? 9);
  };
  const ops = [...(lit.ops || [])].sort(sortByMatch);
  const templates = [...(lit.templates || [])].sort(sortByMatch);

  return (
    <div
      style={{
        background: "var(--bg-tertiary)",
        borderRadius: 6,
        border: "1px solid var(--border)",
        overflow: "hidden",
      }}
    >
      <div style={{ padding: "8px 12px", borderBottom: "1px solid var(--border)" }}>
        <span
          style={{
            fontSize: 12,
            fontWeight: 600,
            color: "var(--text-secondary)",
            textTransform: "uppercase",
          }}
        >
          Literature &amp; Provenance
        </span>
      </div>
      <div style={{ padding: 12, display: "flex", flexDirection: "column", gap: 12 }}>
        {fam && (
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
              <span
                style={{ color: "var(--text-muted)", fontWeight: 600, fontSize: 12, minWidth: 70 }}
              >
                Family:
              </span>
              <span
                style={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary, #e6edf3)" }}
              >
                {fam.family_label}
              </span>
              <MatchBadge matchType={fam.match_type} />
              <ModelLink name={fam.external_model_name} url={fam.reference_url} />
            </div>
            {fam.notes && (
              <div style={{ fontSize: 11, color: "var(--text-muted)", lineHeight: 1.4 }}>
                {fam.notes}
                {fam.citation ? (
                  <span style={{ fontStyle: "italic" }}> — {fam.citation}</span>
                ) : null}
              </div>
            )}
          </div>
        )}
        {!!templates.length && (
          <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
            <span
              style={{
                color: "var(--text-muted)",
                fontWeight: 600,
                fontSize: 11,
                textTransform: "uppercase",
              }}
            >
              Templates
            </span>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {templates.map((t) => (
                <Chip key={t.name} item={t} />
              ))}
            </div>
          </div>
        )}
        {!!ops.length && (
          <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
            <span
              style={{
                color: "var(--text-muted)",
                fontWeight: 600,
                fontSize: 11,
                textTransform: "uppercase",
              }}
            >
              Ops ({ops.length}) — novel first
            </span>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {ops.map((o) => (
                <Chip key={o.name} item={o} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
