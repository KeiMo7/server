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

#include "repository_policy.h"
#include "repository_policy_detail.h"

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <utility>
#include <vector>

namespace triton { namespace server { namespace repository {

using detail::ParseIndexState;
using detail::StabilityGate;

struct RepositoryPolicyEngine::Impl {
  explicit Impl(
      std::string config_name, uint32_t grace, uint32_t force_interval)
      : model_config_name(std::move(config_name)), drain_grace_polls(grace),
        force_poll_interval(force_interval)
  {
  }

  struct VRec {
    VersionState state{VersionState::Unknown};
    std::string reason;
    uint64_t generation{0};
    uint32_t first_seen_poll{0};
    uint32_t last_change_poll{0};
    uint32_t loading_since{0};
    uint32_t unloading_since{0};
    bool error_raised{false};
    bool drain_warn_raised{false};
  };

  struct ModelState {
    VersionPolicy policy;  // effective policy
    bool config_present{false};
    bool config_invalid{false};
    std::string config_error;
    // Initial scan fast-tracks the first fingerprint; later model
    // appearances require two agreeing rounds (the gate is constructed
    // in Observe).
    StabilityGate config_gate{false};
    std::map<uint64_t, VRec> versions;
    std::vector<uint64_t> expected_missing;
    bool ever_seen{false};
    // Guards the no_available_versions warning so it is emitted once per
    // empty-served episode rather than on every poll.
    bool no_versions_warned{false};
  };

  std::string model_config_name;
  uint32_t drain_grace_polls;
  uint32_t force_poll_interval;

  std::map<std::string, ModelState> models;
  // Persistent generation tombstones so a version that disappears and
  // later reappears keeps an increasing generation rather than resetting.
  std::map<std::pair<std::string, uint64_t>, uint64_t> generation_seen;
  // Two-round stability gates keyed independently of the version
  // records, so a directory whose fingerprint is still pending does not
  // appear as a served version.
  std::map<std::pair<std::string, uint64_t>, StabilityGate> fp_gates;

  // Bounded event log built from small immutable chunks. Each Emit
  // appends to the open chunk and only seals it (a shared_ptr swap)
  // when the chunk fills, so an O(1) emission never copies the whole
  // ring. total_event_count_ drives pruning of the oldest chunks to
  // keep the retained set at kEventRingCapacity.
  static constexpr size_t kEventChunkSize = 32;
  std::shared_ptr<EventChunk> open_chunk = std::make_shared<EventChunk>();
  std::shared_ptr<const EventChunk> sealed_chain;
  uint64_t event_seq{0};
  uint64_t total_event_count{0};
  uint64_t discarded_events{0};
  RepositoryCounters counters;
  uint32_t last_full_poll{0};
  bool poll_ever_done{false};
  uint32_t consecutive_index_failures{0};
  bool pending_change{false};
  uint64_t snapshot_seq{0};

  // Publication point. Observe() is single-threaded by contract;
  // readers take the mutex only long enough to copy the pointer, so the
  // immutable snapshot itself is never observed in a torn state.
  mutable std::mutex publish_mu;
  std::shared_ptr<const RepositorySnapshot> published{
      std::make_shared<RepositorySnapshot>()};

  void
  Emit(
      uint32_t poll_index, const std::string& model,
      const std::optional<uint64_t>& version, VersionState from,
      VersionState to, DriftSeverity severity, const std::string& reason)
  {
    RepositoryEvent event;
    event.seq = ++event_seq;
    event.poll_index = poll_index;
    event.model = model;
    event.version = version;
    event.from = from;
    event.to = to;
    event.severity = severity;
    event.reason = Status::SanitizeMessage(reason);

    if (open_chunk->events.size() >= kEventChunkSize) {
      // Seal the full chunk. Snapshots already published keep their own
      // reference, so sealing never mutates an event they can read.
      open_chunk->older = sealed_chain;
      sealed_chain = std::shared_ptr<const EventChunk>(open_chunk);
      open_chunk = std::make_shared<EventChunk>();
    }
    open_chunk->events.push_back(std::move(event));
    ++total_event_count;
    PruneOldEvents();

    if (severity == DriftSeverity::Error) {
      ++counters.drift_error;
    } else if (severity == DriftSeverity::Warn) {
      ++counters.drift_warn;
    }
  }

  // Drop whole sealed chunks from the oldest end while the retained
  // event count exceeds the ring capacity. Sealed chunks are never
  // mutated; the open (newest) chunk is always retained, so its events
  // survive even under a burst of emissions.
  void
  PruneOldEvents()
  {
    while (total_event_count > RepositorySnapshot::kEventRingCapacity &&
           sealed_chain) {
      // Walk to the oldest chunk, which has a null 'older' link. Track
      // its predecessor so it can be unlinked.
      std::shared_ptr<const EventChunk> prev;
      std::shared_ptr<const EventChunk> cur = sealed_chain;
      while (cur->older) {
        prev = cur;
        cur = cur->older;
      }
      total_event_count -= cur->events.size();
      discarded_events += cur->events.size();
      if (prev) {
        std::const_pointer_cast<EventChunk>(prev)->older.reset();
      } else {
        sealed_chain.reset();
      }
    }
  }

  void
  ChangeState(
      uint32_t poll_index, const std::string& model, uint64_t version,
      VRec* rec, VersionState to, const std::string& reason,
      DriftSeverity severity = DriftSeverity::None)
  {
    VersionState from = rec->state;
    if (from == to && rec->reason == reason) {
      return;
    }
    rec->state = to;
    rec->reason = reason;
    rec->last_change_poll = poll_index;
    rec->generation = ++generation_seen[std::make_pair(model, version)];
    if (rec->first_seen_poll == 0) {
      rec->first_seen_poll = poll_index;
    }
    if (to == VersionState::Loading) {
      if (from != VersionState::Loading) {
        rec->loading_since = poll_index;
      }
      rec->error_raised = false;
      ++counters.load_expected;
    } else if (to == VersionState::Unloading) {
      if (rec->unloading_since == 0) {
        rec->unloading_since = poll_index;
      }
      ++counters.unload_expected;
    } else if (to == VersionState::Ready) {
      rec->loading_since = 0;
      rec->error_raised = false;
    }
    Emit(poll_index, model, version, from, to, severity, reason);
  }

  // Publish an immutable snapshot assembled from current state.
  void
  Publish(uint32_t poll_index)
  {
    auto snap = std::make_shared<RepositorySnapshot>();
    snap->seq = ++snapshot_seq;
    snap->poll_index = poll_index;
    snap->discarded_events = discarded_events;
    snap->counters = counters;
    snap->models.reserve(models.size());

    for (const auto& mpair : models) {
      ModelView view;
      view.name = mpair.first;
      view.policy = mpair.second.policy;
      view.expected_missing = mpair.second.expected_missing;
      for (const auto& vpair : mpair.second.versions) {
        const VRec& rec = vpair.second;
        ModelVersionView vv;
        vv.version = vpair.first;
        vv.state = rec.state;
        vv.reason = rec.reason;
        vv.generation = rec.generation;
        vv.first_seen_poll = rec.first_seen_poll;
        vv.last_change_poll = rec.last_change_poll;
        if (rec.state == VersionState::Unloading) {
          view.draining.push_back(vpair.first);
        } else {
          view.served.push_back(std::move(vv));
        }
      }
      // std::map keys already give ascending model and version order.
      snap->models.push_back(std::move(view));
    }

    // Publish the current chain. Sealed chunks are immutable and are
    // shared by reference; only the small open chunk is copied, so a
    // later Emit (which seals before appending) cannot mutate an event
    // reachable from this snapshot.
    if (open_chunk->events.empty()) {
      snap->events = sealed_chain;
    } else {
      auto view_chunk = std::make_shared<EventChunk>();
      view_chunk->events = open_chunk->events;  // copy at most kEventChunkSize
      view_chunk->older = sealed_chain;
      snap->events = std::shared_ptr<const EventChunk>(view_chunk);
    }
    std::lock_guard<std::mutex> lock(publish_mu);
    published = std::move(snap);
  }
};

RepositoryPolicyEngine::RepositoryPolicyEngine(
    std::string model_config_name, uint32_t drain_grace_polls,
    uint32_t force_poll_interval)
    : impl_(new Impl(
          std::move(model_config_name), drain_grace_polls,
          force_poll_interval))
{
}

RepositoryPolicyEngine::~RepositoryPolicyEngine() = default;

void
RepositoryPolicyEngine::Observe(
    const RepositoryScan& scan,
    const std::optional<IndexSnapshot>& index, uint32_t poll_index)
{
  Impl& st = *impl_;
  const bool have_index = index.has_value();
  if (have_index) {
    st.consecutive_index_failures = 0;
  } else {
    ++st.consecutive_index_failures;
    st.Emit(
        poll_index, std::string(), std::nullopt, VersionState::Unknown,
        VersionState::Unknown, DriftSeverity::Warn, "index_unavailable");
  }

  // Index rows grouped by model; rows without a version describe the
  // model itself and are tracked separately.
  std::map<std::string, std::map<uint64_t, IndexEntry>> index_by_model;
  std::set<std::string> index_model_names;
  if (have_index) {
    for (const auto& entry : index->entries) {
      index_model_names.insert(entry.name);
      if (entry.version.has_value()) {
        index_by_model[entry.name].emplace(entry.version.value(), entry);
      }
    }
  }

  std::set<std::string> disk_models;
  st.pending_change = false;

  for (const ModelScan& ms : scan.models) {
    disk_models.insert(ms.name);
    Impl::ModelState& mstate = st.models[ms.name];
    mstate.ever_seen = true;

    // ---- Configuration: two-round stability on the fingerprint.
    bool config_unstable = false;
    uint64_t effective_config_fp =
        mstate.config_gate.Update(ms.config_fingerprint, &config_unstable);
    if (effective_config_fp != ms.config_fingerprint) {
      st.pending_change = true;  // new config held until confirmed
    }
    if (config_unstable) {
      st.Emit(
          poll_index, ms.name, std::nullopt, VersionState::Unknown,
          VersionState::Unknown, DriftSeverity::Warn, "unstable_repository");
    }

    // Only parse configuration whose fingerprint is currently effective.
    VersionPolicy parsed_policy;
    bool config_invalid = false;
    std::string config_error;
    if (ms.config_present && effective_config_fp == ms.config_fingerprint) {
      Status ps = ParseVersionPolicy(ms.config_text, &parsed_policy);
      if (!ps.ok) {
        config_invalid = true;
        config_error = ps.message;
      }
    } else if (ms.config_present) {
      parsed_policy = mstate.policy;  // keep prior policy while pending
    }
    mstate.config_present = ms.config_present;
    mstate.config_invalid = config_invalid;
    mstate.config_error = config_error;
    if (!config_invalid) {
      mstate.policy = parsed_policy;
    }

    // ---- Version fingerprints: two-round stability per directory.
    std::map<uint64_t, const VersionEntry*> effective_entries;
    for (const VersionEntry& ve : ms.versions) {
      auto gate_key = std::make_pair(ms.name, ve.version);
      auto gate_it = st.fp_gates.find(gate_key);
      if (gate_it == st.fp_gates.end()) {
        // After the first completed scan, newly appearing directories
        // must wait for one confirming round.
        gate_it = st.fp_gates
                      .emplace(
                          gate_key, StabilityGate(st.poll_ever_done))
                      .first;
      }
      StabilityGate& gate = gate_it->second;
      bool unstable = false;
      uint64_t effective_fp = gate.Update(ve.fingerprint, &unstable);
      if (unstable) {
        st.Emit(
            poll_index, ms.name, ve.version, VersionState::Unknown,
            VersionState::Unknown, DriftSeverity::Warn,
            "unstable_repository");
      }
      if (effective_fp != ve.fingerprint) {
        st.pending_change = true;
      } else {
        // Effective: make sure a migration record exists and expose the
        // candidate to policy application.
        if (mstate.versions.count(ve.version) == 0) {
          Impl::VRec rec;
          mstate.versions.emplace(ve.version, std::move(rec));
        }
        effective_entries.emplace(ve.version, &ve);
      }
    }

    // ---- Expected set from the (effective) policy.
    std::vector<VersionEntry> candidates;
    candidates.reserve(effective_entries.size());
    for (const auto& epair : effective_entries) {
      candidates.push_back(*epair.second);
    }
    std::vector<uint64_t> missing;
    std::vector<uint64_t> expected;
    if (config_invalid) {
      mstate.expected_missing.clear();
    } else {
      expected = ApplyVersionPolicy(mstate.policy, candidates, &missing);
      mstate.expected_missing = missing;
      // Non-strict runtime signal for a model with nothing to serve
      // (e.g. a specific policy whose versions are all absent). Warned
      // once until the model regains at least one served version.
      if (expected.empty() && have_index && !mstate.no_versions_warned) {
        st.Emit(
            poll_index, ms.name, std::nullopt, VersionState::Unknown,
            VersionState::Unknown, DriftSeverity::Warn,
            "no_available_versions");
        mstate.no_versions_warned = true;
      } else if (!expected.empty()) {
        mstate.no_versions_warned = false;
      }
    }
    std::set<uint64_t> expected_set(expected.begin(), expected.end());

    const auto* idx_rows =
        have_index ? &index_by_model[ms.name] : nullptr;

    // ---- Reconcile every version still tracked for this model.
    std::vector<uint64_t> erase_versions;
    for (auto& vpair : mstate.versions) {
      uint64_t version = vpair.first;
      Impl::VRec& rec = vpair.second;
      bool on_disk = expected_set.count(version) != 0;
      const IndexEntry* idx = nullptr;
      if (idx_rows != nullptr) {
        auto found = idx_rows->find(version);
        if (found != idx_rows->end()) {
          idx = &found->second;
        }
      }

      if (config_invalid) {
        st.ChangeState(
            poll_index, ms.name, version, &rec, VersionState::Unavailable,
            config_error, DriftSeverity::Error);
        continue;
      }

      if (!on_disk) {
        // Removed from disk, or version directory change still pending.
        if (idx != nullptr) {
          st.ChangeState(
              poll_index, ms.name, version, &rec, VersionState::Unloading,
              "version_removed", DriftSeverity::None);
          ++st.counters.drain_observed;
          if (rec.drain_warn_raised == false &&
              (poll_index - rec.unloading_since) > st.drain_grace_polls) {
            st.Emit(
                poll_index, ms.name, version, VersionState::Unloading,
                VersionState::Unloading, DriftSeverity::Warn,
                "drain_grace_exceeded");
            rec.drain_warn_raised = true;
          }
        } else {
          // Gone from both disk and index: retain nothing but keep the
          // generation tombstone for a possible reappearance.
          erase_versions.push_back(version);
        }
        continue;
      }

      // On-disk expected version.
      if (!have_index) {
        // Index fetch failed: refresh disk-side expectation but do not
        // advance any version state.
        if (rec.state == VersionState::Unknown) {
          rec.reason = "awaiting_index";
        }
        continue;
      }

      if (idx == nullptr) {
        // Expected on disk but absent from the index: loading drift.
        st.ChangeState(
            poll_index, ms.name, version, &rec, VersionState::Loading,
            "load_expected", DriftSeverity::None);
        if (!rec.error_raised &&
            (poll_index - rec.loading_since) > 2) {
          st.Emit(
              poll_index, ms.name, version, VersionState::Loading,
              VersionState::Loading, DriftSeverity::Error,
              "expected_version_missing");
          rec.error_raised = true;
        }
        continue;
      }

      VersionState index_state = ParseIndexState(idx->state);
      if (index_state == VersionState::Unavailable) {
        // Keep the index's own reason text verbatim.
        st.ChangeState(
            poll_index, ms.name, version, &rec, VersionState::Unavailable,
            idx->reason.empty() ? std::string("unavailable") : idx->reason,
            DriftSeverity::Error);
        continue;
      }
      st.ChangeState(
          poll_index, ms.name, version, &rec, index_state, idx->reason);
    }
    for (uint64_t v : erase_versions) {
      mstate.versions.erase(v);
      st.fp_gates.erase(std::make_pair(ms.name, v));
    }
  }

  // ---- Models present in the index but absent from disk.
  if (have_index) {
    std::vector<std::string> candidate_names;
    candidate_names.reserve(index_model_names.size());
    for (const auto& mpair : st.models) {
      if (disk_models.count(mpair.first) == 0 &&
          !mpair.second.ever_seen) {
        candidate_names.push_back(mpair.first);
      }
    }
    for (const auto& name : index_model_names) {
      if (disk_models.count(name) == 0) {
        candidate_names.push_back(name);
      }
    }
    std::sort(candidate_names.begin(), candidate_names.end());
    candidate_names.erase(
        std::unique(candidate_names.begin(), candidate_names.end()),
        candidate_names.end());

    for (const auto& name : candidate_names) {
      Impl::ModelState& mstate = st.models[name];
      const auto rows_it = index_by_model.find(name);
      std::set<uint64_t> live_index_versions;
      if (rows_it != index_by_model.end()) {
        for (const auto& rpair : rows_it->second) {
          live_index_versions.insert(rpair.first);
          auto it = mstate.versions.find(rpair.first);
          if (it == mstate.versions.end()) {
            Impl::VRec rec;
            rec.state = VersionState::Unknown;
            rec.reason = "index_only";
            rec.generation =
                ++st.generation_seen[std::make_pair(name, rpair.first)];
            rec.first_seen_poll = poll_index;
            rec.last_change_poll = poll_index;
            mstate.versions.emplace(rpair.first, std::move(rec));
          } else if (it->second.state == VersionState::Unknown) {
            it->second.reason = "index_only";
          }
        }
      }
      // Versions the index no longer reports are dropped so the model
      // resolves by itself on the next index update.
      std::vector<uint64_t> stale;
      for (const auto& vpair : mstate.versions) {
        if (live_index_versions.count(vpair.first) == 0) {
          stale.push_back(vpair.first);
        }
      }
      for (uint64_t v : stale) {
        mstate.versions.erase(v);
        st.fp_gates.erase(std::make_pair(name, v));
      }
      // An index-only model the index itself no longer mentions is
      // removed entirely.
      if (mstate.versions.empty()) {
        st.models.erase(name);
      }
    }
  }

  st.last_full_poll = poll_index;
  st.poll_ever_done = true;
  st.Publish(poll_index);
}

void
RepositoryPolicyEngine::ObserveScanFailure(
    const Status& status, uint32_t poll_index,
    const std::string& repository_path)
{
  Impl& st = *impl_;
  ++st.counters.scan_failure;
  st.Emit(
      poll_index, repository_path, std::nullopt, VersionState::Unknown,
      VersionState::Unknown,
      st.counters.scan_failure >= 3 ? DriftSeverity::Error
                                   : DriftSeverity::Warn,
      status.ok ? "scan_failure" : status.message);
  // Snapshot is republished but all model state is retained unchanged.
  st.Publish(poll_index);
}

bool
RepositoryPolicyEngine::DecidePoll(uint32_t poll_index) const
{
  const Impl& st = *impl_;
  if (!st.poll_ever_done) {
    return true;
  }
  if (st.consecutive_index_failures >= 3) {
    return true;
  }
  if (st.pending_change) {
    return true;
  }
  // Interval cap so an undetected change can never persist past
  // force_poll_interval poll intervals.
  if ((poll_index - st.last_full_poll) >= st.force_poll_interval) {
    return true;
  }
  return false;
}

void
RepositoryPolicyEngine::NoteSkippedPoll()
{
  ++impl_->counters.skipped_polls;
}

std::shared_ptr<const RepositorySnapshot>
RepositoryPolicyEngine::Snapshot() const
{
  std::lock_guard<std::mutex> lock(impl_->publish_mu);
  return impl_->published;
}

RepositoryCounters
RepositoryPolicyEngine::Counters() const
{
  std::lock_guard<std::mutex> lock(impl_->publish_mu);
  return impl_->published->counters;
}

std::vector<RepositoryEvent>
RepositoryPolicyEngine::Events() const
{
  std::lock_guard<std::mutex> lock(impl_->publish_mu);
  return FlattenEventChunks(impl_->published->events);
}

}}}  // namespace triton::server::repository
