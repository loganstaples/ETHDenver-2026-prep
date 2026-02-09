//! Prometheus-compatible metrics export for the HELIX node API.
//!
//! Converts internal `MetricsSnapshot` into Prometheus text exposition format
//! so external monitoring tools (Grafana, Prometheus, etc.) can scrape node metrics.

use super::http::MetricsSnapshot;

/// Renders a `MetricsSnapshot` as Prometheus text exposition format.
///
/// Example output:
/// ```text
/// # HELP helix_rounds_total Total training rounds completed.
/// # TYPE helix_rounds_total counter
/// helix_rounds_total 42
/// ```
pub fn to_prometheus(snapshot: &MetricsSnapshot) -> String {
    let mut out = String::with_capacity(512);

    push_counter(&mut out, "helix_rounds_total", "Total training rounds completed", snapshot.total_rounds);
    push_counter(&mut out, "helix_proofs_total", "Total ZK proofs verified", snapshot.total_proofs);
    push_counter(&mut out, "helix_proofs_valid_total", "ZK proofs that passed verification", snapshot.proofs_valid);
    push_counter(&mut out, "helix_proofs_invalid_total", "ZK proofs that failed verification", snapshot.proofs_invalid);
    push_gauge_f64(&mut out, "helix_avg_round_time_ms", "Average round time in milliseconds", snapshot.avg_round_time_ms);
    push_gauge(&mut out, "helix_uptime_seconds", "Node uptime in seconds", snapshot.uptime_secs);

    out
}

fn push_counter(out: &mut String, name: &str, help: &str, value: u64) {
    out.push_str(&format!(
        "# HELP {} {}\n# TYPE {} counter\n{} {}\n",
        name, help, name, name, value
    ));
}

fn push_gauge(out: &mut String, name: &str, help: &str, value: u64) {
    out.push_str(&format!(
        "# HELP {} {}\n# TYPE {} gauge\n{} {}\n",
        name, help, name, name, value
    ));
}

fn push_gauge_f64(out: &mut String, name: &str, help: &str, value: f64) {
    out.push_str(&format!(
        "# HELP {} {}\n# TYPE {} gauge\n{} {:.3}\n",
        name, help, name, name, value
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prometheus_format() {
        let snapshot = MetricsSnapshot {
            total_rounds: 10,
            total_proofs: 30,
            proofs_valid: 28,
            proofs_invalid: 2,
            avg_round_time_ms: 1500.5,
            uptime_secs: 3600,
        };

        let output = to_prometheus(&snapshot);
        assert!(output.contains("helix_rounds_total 10"));
        assert!(output.contains("helix_proofs_total 30"));
        assert!(output.contains("helix_proofs_valid_total 28"));
        assert!(output.contains("helix_proofs_invalid_total 2"));
        assert!(output.contains("helix_uptime_seconds 3600"));
        assert!(output.contains("# TYPE helix_rounds_total counter"));
        assert!(output.contains("# TYPE helix_uptime_seconds gauge"));
    }
}
