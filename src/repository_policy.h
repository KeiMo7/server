// Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Backend-independent repository policy and reconciliation module.
//
// This module decides *which* model versions should be served from a
// repository scan according to the text version_policy configuration,
// reconciles that expectation with the index reported by
// libtritonserver, and produces an immutable snapshot plus a bounded
// stream of events. The module never loads or unloads anything itself:
// the caller (src/main.cc, or the repository_policy_check tool) drives
// TRITONSERVER_ServerPollModelRepository and converts the resulting
// index into IndexSnapshot before calling Observe().
//
// No tritonserver.h or backend headers are included here, and the
// module reads no clocks and starts no threads: every time-dependent
// decision is driven by the poll_index supplied by the caller.
#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace triton { namespace server { namespace repository {

// Mirrors TRITONSERVER_ModelControlMode ordering (NONE, POLL,
// EXPLICIT). Defined locally so this module does not depend on
// tritonserver.h.
enum class ControlMode { None = 0, Poll = 1, Explicit = 2 };

enum class PolicyKind { All, Latest, Specific };

// Internal migration state. The values map onto the fixed set of index
// state strings when exposed externally (see IndexStateName); no new
// state string is ever emitted.
enum class VersionState {
  Unknown,
  Loading,
  Ready,
  Unavailable,
  Unloading
};

enum class DriftSeverity { None, Warn, Error };

// Text form of a version policy, as parsed from the version_policy
// block of a config.pbtxt (text protobuf). 'num_versions' is only
// meaningful for Latest and 'versions' is only meaningful for
// Specific.
struct VersionPolicy {
  PolicyKind kind{PolicyKind::Latest};
  uint32_t num_versions{1};
  std::vector<uint64_t> versions;

  bool
  operator==(const VersionPolicy& other) const
  {
    return kind == other.kind && num_versions == other.num_versions &&
           versions == other.versions;
  }
  bool
  operator!=(const VersionPolicy& other) const
  {
    return !(*this == other);
  }
};

// One candidate version directory discovered on disk.
struct VersionEntry {
  uint64_t version{0};
  // Path relative to the repository root, e.g. "model_a/3".
  std::string relative_path;
  uint64_t file_count{0};
  uint64_t total_bytes{0};
  // FNV-1a 64-bit digest synthesised from file_count, total_bytes and
  // the greatest last_write_time (nanoseconds) of the regular files.
  uint64_t fingerprint{0};
};

struct ModelScan {
  std::string name;
  bool config_present{false};
  // FNV-1a 64-bit digest of the resolved configuration file bytes.
  uint64_t config_fingerprint{0};
  std::vector<VersionEntry> versions;
  // Raw numeric values parsed from version directory names that had to
  // be skipped (leading zeros, zero, overflow, ...). Non-numeric names
  // have no numeric representation and therefore cannot appear here.
  std::vector<uint64_t> skipped;
  // Raw bytes of the resolved configuration file when config_present is
  // true; empty otherwise. The engine parses the version_policy block
  // only after the fingerprint passes its two-round stability gate.
  std::string config_text;
};

struct RepositoryScan {
  uint64_t scan_seq{0};
  std::vector<ModelScan> models;
};

// One row of /v2/repository/index after the caller has converted it.
struct IndexEntry {
  std::string name;
  std::optional<uint64_t> version;
  std::string state;
  std::string reason;
};

struct IndexSnapshot {
  std::vector<IndexEntry> entries;
};

struct ModelVersionView {
  uint64_t version{0};
  VersionState state{VersionState::Unknown};
  std::string reason;
  uint64_t generation{0};
  uint32_t first_seen_poll{0};
  uint32_t last_change_poll{0};
};

struct ModelView {
  std::string name;
  VersionPolicy policy;
  // Sorted ascending by numeric version.
  std::vector<ModelVersionView> served;
  // Versions requested by a Specific policy that have no candidate
  // directory on disk; sorted ascending.
  std::vector<uint64_t> expected_missing;
  // Versions that disappeared from disk but are still referenced by
  // the index (draining in-flight requests); sorted ascending.
  std::vector<uint64_t> draining;
};

struct RepositoryEvent {
  uint64_t seq{0};
  uint32_t poll_index{0};
  std::string model;
  std::optional<uint64_t> version;
  VersionState from{VersionState::Unknown};
  VersionState to{VersionState::Unknown};
  DriftSeverity severity{DriftSeverity::None};
  std::string reason;
};

struct RepositoryCounters {
  uint64_t load_expected{0};
  uint64_t unload_expected{0};
  uint64_t drift_error{0};
  uint64_t drift_warn{0};
  uint64_t drain_observed{0};
  uint64_t scan_failure{0};
  uint64_t skipped_polls{0};
};

// One immutable chunk of the bounded event log. Chunks are shared
// between snapshots, so publishing a snapshot that contains new events
// only copies the events in the newest chunk; older chunks are
// referenced, not copied.
struct EventChunk {
  std::vector<RepositoryEvent> events;
  std::shared_ptr<const EventChunk> older;
};

// Flatten an event chunk chain into one oldest-to-newest vector.
// Provided so callers that need a plain vector (tests, JSON output) do
// not have to know about the chunk layout.
std::vector<RepositoryEvent>
FlattenEventChunks(const std::shared_ptr<const EventChunk>& newest);

// Immutable point-in-time view of the whole repository. Published
// atomically; readers always observe a fully consistent object.
struct RepositorySnapshot {
  uint64_t seq{0};
  uint32_t poll_index{0};
  // Sorted ascending by model name; versions sorted numerically.
  std::vector<ModelView> models;
  // Newest chunk of the bounded event ring; walk 'older' to read the
  // rest in order. The newest kEventRingCapacity events are retained.
  std::shared_ptr<const EventChunk> events;
  // Number of events discarded because the ring was full.
  uint64_t discarded_events{0};
  RepositoryCounters counters;

  static constexpr size_t kEventRingCapacity = 1024;
};

// Status used by the free-standing scan / parse helpers. The engine's
// Observe() never throws and reports failures through events instead.
struct Status {
  bool ok{true};
  std::string message;

  static Status
  OK()
  {
    return Status{true, std::string()};
  }
  static Status
  Error(const std::string& msg)
  {
    return Status{false, SanitizeMessage(msg)};
  }

  // Truncate to 256 bytes and strip line breaks so an error string can
  // never break a single-line log entry.
  static std::string SanitizeMessage(const std::string& in);
};

// Fixed index state vocabulary (extension_model_repository.md).
const char*
IndexStateName(VersionState state);
const char*
SeverityName(DriftSeverity severity);
const char*
PolicyKindName(PolicyKind kind);

// Parse a version_policy block out of configuration text. Only the
// version_policy block is interpreted; surrounding configuration is
// ignored. Accepts all{} / latest{num_versions:n} /
// specific{versions:[1,3]}, tolerates arbitrary whitespace, '#' line
// comments and several fields on one line. A malformed block yields a
// non-OK Status whose message carries the first offending position.
Status
ParseVersionPolicy(const std::string& config_text, VersionPolicy* policy);

// Apply a parsed policy to candidate versions (any order) and return
// the ascending set of versions that should be served. 'missing'
// receives Specific versions with no candidate, also ascending.
std::vector<uint64_t>
ApplyVersionPolicy(
    const VersionPolicy& policy,
    const std::vector<VersionEntry>& candidates,
    std::vector<uint64_t>* missing);

// Result of routing one inference request to a model's versions.
struct RouteResult {
  bool ok{false};
  uint64_t version{0};
  std::string error;
};

// Route a request's version string against a model view. An empty
// request selects the greatest served version. A non-numeric request
// is an "invalid version" error. A numeric version that is not served
// produces the same wording as the server's existing unknown-version
// responses.
RouteResult
RouteVersion(const ModelView& view, const std::string& requested);

// Render one event as a single deterministic log line:
// "repo-policy: model=.. version=.. from=.. to=.. severity=.. reason=.."
// Fields are always present in that fixed order.
std::string
FormatEventLine(const RepositoryEvent& event);

// Driven exclusively from the single writer thread that owns the
// repository polling loop. Readers may call the const accessors from
// any thread.
class RepositoryPolicyEngine {
 public:
  // 'drain_grace_polls' is the number of poll intervals a version may
  // stay in the index after disappearing from disk before the drift is
  // escalated to a (Warn-only) event. 'force_poll_interval' bounds the
  // number of intervals between forced full polls even when no change
  // signature is observed.
  RepositoryPolicyEngine(
      std::string model_config_name = {}, uint32_t drain_grace_polls = 2,
      uint32_t force_poll_interval = 8);
  ~RepositoryPolicyEngine();

  RepositoryPolicyEngine(const RepositoryPolicyEngine&) = delete;
  RepositoryPolicyEngine&
  operator=(const RepositoryPolicyEngine&) = delete;

  // Reconcile one disk scan with an optional index snapshot and
  // publish a new immutable snapshot. A nullopt index means the index
  // could not be fetched: disk-side expectations are refreshed but no
  // version state advances and no Error drift is raised. Never throws.
  void
  Observe(
      const RepositoryScan& scan,
      const std::optional<IndexSnapshot>& index, uint32_t poll_index);

  // Observe() variant for a scan that itself failed. The previous
  // snapshot and all migration state are retained; a Warn event is
  // appended and scan_failure is incremented.
  void
  ObserveScanFailure(
      const Status& status, uint32_t poll_index,
      const std::string& repository_path);

  // Whether the caller should actually invoke
  // TRITONSERVER_ServerPollModelRepository on this interval. Returns
  // true when a change signature is pending, after index failures, or
  // unconditionally once force_poll_interval intervals elapsed since
  // the last full poll, so a missed change can never persist.
  bool
  DecidePoll(uint32_t poll_index) const;

  // Notify that the caller skipped a poll (DecidePoll() == false).
  // Only bookkeeping: the skipped_polls counter is incremented.
  void
  NoteSkippedPoll();

  // Immutable snapshot accessors. Each returns a private copy (for the
  // snapshot, a shared_ptr to an immutable object); internal
  // containers are never exposed.
  std::shared_ptr<const RepositorySnapshot>
  Snapshot() const;
  RepositoryCounters
  Counters() const;
  std::vector<RepositoryEvent>
  Events() const;

 private:
  struct Impl;
  const std::unique_ptr<Impl> impl_;
};

}}}  // namespace triton::server::repository
