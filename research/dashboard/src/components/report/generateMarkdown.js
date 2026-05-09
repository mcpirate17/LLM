export default function generateMarkdown(data) {
  const s = data.summary || {};
  const s1Survivors = s.stage1_survivors ?? s.total_s1_passed ?? 0;
  const lines = [];
  lines.push('# Research Report');
  lines.push(`*Generated: ${new Date().toISOString()}*\n`);

  if (data.narrative) {
    lines.push('## Executive Summary\n');
    lines.push(data.narrative + '\n');
  }

  lines.push('## Key Statistics\n');
  lines.push(`- Total experiments: ${s.total_experiments || 0}`);
  lines.push(`- Programs evaluated: ${s.total_programs_evaluated || 0}`);
  lines.push(`- Stage 1 survivors: ${s1Survivors}`);
  lines.push(`- Novel discoveries: ${s.total_novel || 0}`);
  lines.push('');

  const top = data.top_programs || [];
  if (top.length > 0) {
    lines.push('## Discovery Rankings\n');
    lines.push('| Rank | Fingerprint | VAL Loss Ratio | Novelty | Baseline | Similar To |');
    lines.push('|------|-------------|-----------------------|---------|----------|------------|');
    top.forEach((p, i) => {
      const loss = p.validation_loss_ratio != null ? p.validation_loss_ratio : p.loss_ratio;
      lines.push(
        `| ${i + 1} | \`${(p.graph_fingerprint || '').slice(0, 12)}\` ` +
        `| ${loss != null ? loss.toFixed(4) : '--'} ` +
        `| ${p.novelty_score != null ? p.novelty_score.toFixed(3) : '--'} ` +
        `| ${p.baseline_loss_ratio != null ? p.baseline_loss_ratio.toFixed(3) : '--'} ` +
        `| ${p.most_similar_to || '--'} |`
      );
    });
    lines.push('');
  }

  const ops = data.op_success_rates || [];
  if (ops.length > 0) {
    lines.push('## Op Success Rates\n');
    lines.push('| Op | S1 Rate | Count |');
    lines.push('|----|---------|-------|');
    (Array.isArray(ops) ? ops : []).slice(0, 20).forEach(op => {
      lines.push(`| ${op.op_name || '?'} | ${op.s1_rate != null ? (op.s1_rate * 100).toFixed(1) + '%' : '--'} | ${op.total_count || '--'} |`);
    });
    lines.push('');
  }

  const failures = data.failure_patterns || {};
  if (Object.keys(failures).length > 0) {
    lines.push('## Failure Patterns\n');
    lines.push('```json');
    lines.push(JSON.stringify(failures, null, 2));
    lines.push('```\n');
  }

  const insights = data.insights || [];
  if (insights.length > 0) {
    lines.push('## Insights\n');
    insights.forEach(ins => {
      lines.push(`- **[${ins.category || 'general'}]** ${ins.content || ins}`);
    });
    lines.push('');
  }

  return lines.join('\n');
}
