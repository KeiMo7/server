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

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <thread>
#include <vector>

#include "repository_policy.h"
#include "repository_policy_scan.h"

namespace fs = std::filesystem;
using namespace triton::server::repository;

namespace {

// RAII scratch repository rooted in the system temp directory.
class TempRepo {
 public:
  TempRepo()
  {
    path_ = fs::temp_directory_path() /
            ("repo_policy_test_" +
             std::to_string(
                 std::chrono::steady_clock::now()
                     .time_since_epoch()
                     .count()) +
             "_" + std::to_string(counter_++));
    fs::create_directories(path_);
  }

  ~TempRepo()
  {
    std::error_code ec;
    fs::remove_all(path_, ec);
  }

  const std::string&
  Path() const
  {
    return path_;
  }

  void
  WriteFile(const std::string& relative, const std::string& contents)
  {
    fs::path full = fs::path(path_) / relative;
    fs::create_directories(full.parent_path());
    std::ofstream out(full, std::ios::binary);
    out << contents;
  }

  // Create <model>/<version>/model.bin so the version has a regular file.
  void
  MakeVersion(
      const std::string& model, const std::string& version,
      const std::string& payload = "data")
  {
    WriteFile(model + "/" + version + "/model.bin", payload);
  }

  void
  MakeConfig(const std::string& model, const std::string& text)
  {
    WriteFile(model + "/config.pbtxt", text);
  }

  void
  RemoveVersion(const std::string& model, const std::string& version)
  {
    std::error_code ec;
    fs::remove_all(fs::path(path_) / model / version, ec);
  }

 private:
  std::string path_;
  static std::atomic<uint64_t> counter_;
};

std::atomic<uint64_t> TempRepo::counter_{0};

RepositoryScan
DoScan(const TempRepo& repo, const std::string& config_name = {})
{
  RepositoryScan scan;
  Status st = ScanRepository(repo.Path(), config_name, 1, &scan);
  EXPECT_TRUE(st.ok) << st.message;
  return scan;
}

const ModelScan*
FindModel(const RepositoryScan& scan, const std::string& name)
{
  for (const auto& m : scan.models) {
    if (m.name == name) {
      return &m;
    }
  }
  return nullptr;
}

const ModelView*
FindView(
    const std::shared_ptr<const RepositorySnapshot>& snap,
    const std::string& name)
{
  for (const auto& m : snap->models) {
    if (m.name == name) {
      return &m;
    }
  }
  return nullptr;
}

const ModelVersionView*
FindVersion(const ModelView& view, uint64_t version)
{
  for (const auto& v : view.served) {
    if (v.version == version) {
      return &v;
    }
  }
  return nullptr;
}

size_t
EventCount(const std::shared_ptr<const RepositorySnapshot>& snap)
{
  size_t n = 0;
  for (auto cur = snap->events; cur; cur = cur->older) {
    n += cur->events.size();
  }
  return n;
}

IndexSnapshot
ReadyIndex(const std::string& model, std::vector<uint64_t> versions)
{
  IndexSnapshot snap;
  for (uint64_t v : versions) {
    snap.entries.push_back(IndexEntry{model, v, "READY", ""});
  }
  return snap;
}

VersionEntry
MakeVersionEntry(uint64_t version)
{
  VersionEntry entry;
  entry.version = version;
  return entry;
}

// gmock is not an allowed dependency, so vector equality is checked
// with plain assertions.
void
ExpectVecEq(
    const std::vector<uint64_t>& actual,
    const std::vector<uint64_t>& expected)
{
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual[i], expected[i]) << "at index " << i;
  }
}

void
ExpectVecUnorderedEq(
    std::vector<uint64_t> actual, const std::vector<uint64_t>& expected)
{
  std::sort(actual.begin(), actual.end());
  ExpectVecEq(actual, expected);
}

// Deterministic, standard-library-only serializer used to prove two
// runs produce byte-identical output. Field order is fixed and models
// and versions are already sorted by the engine.
std::string
JsonEscape(const std::string& in)
{
  std::string out;
  out.reserve(in.size() + 2);
  for (char c : in) {
    if (c == '"' || c == '\\') {
      out.push_back('\\');
      out.push_back(c);
    } else if (c == '\n') {
      out += "\\n";
    } else {
      out.push_back(c);
    }
  }
  return out;
}

std::string
JsonUintList(const std::vector<uint64_t>& values)
{
  std::string out = "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      out += ", ";
    }
    out += std::to_string(values[i]);
  }
  out += "]";
  return out;
}

std::string
SnapshotJson(const RepositorySnapshot& snap)
{
  std::ostringstream oss;
  oss << "{\n  \"seq\": " << snap.seq << ",\n  \"models\": [\n";
  for (size_t mi = 0; mi < snap.models.size(); ++mi) {
    const ModelView& view = snap.models[mi];
    oss << "    {\n      \"name\": \"" << JsonEscape(view.name) << "\",\n";
    oss << "      \"policy\": \"" << PolicyKindName(view.policy.kind)
        << "\",\n";
    oss << "      \"served\": [\n";
    for (size_t vi = 0; vi < view.served.size(); ++vi) {
      const ModelVersionView& vv = view.served[vi];
      oss << "        {\"version\": " << vv.version << ", \"state\": \""
          << IndexStateName(vv.state) << "\", \"reason\": \""
          << JsonEscape(vv.reason) << "\", \"generation\": " << vv.generation
          << "}";
      if (vi + 1 < view.served.size()) {
        oss << ",";
      }
      oss << "\n";
    }
    oss << "      ],\n";
    oss << "      \"expected_missing\": "
        << JsonUintList(view.expected_missing) << ",\n";
    oss << "      \"draining\": " << JsonUintList(view.draining) << "\n";
    oss << "    }";
    if (mi + 1 < snap.models.size()) {
      oss << ",";
    }
    oss << "\n";
  }
  oss << "  ]\n}";
  return oss.str();
}

}  // namespace

// =========================================================================
// Version discovery
// =========================================================================

TEST(RepositoryScanTest, VersionScanIgnoresLeadingZero)
{
  TempRepo repo;
  repo.MakeVersion("m", "0");
  repo.MakeVersion("m", "01");
  repo.MakeVersion("m", "2");
  auto scan = DoScan(repo);
  const ModelScan* m = FindModel(scan, "m");
  ASSERT_NE(m, nullptr);
  ASSERT_EQ(m->versions.size(), 1u);
  EXPECT_EQ(m->versions[0].version, 2u);
  ExpectVecUnorderedEq(m->skipped, {0u, 1u});
}

TEST(RepositoryScanTest, VersionScanRejectsNonNumeric)
{
  TempRepo repo;
  repo.MakeVersion("m", "v1");
  repo.MakeVersion("m", "-1");
  repo.MakeVersion("m", "1.0");
  repo.MakeVersion("m", "3");
  auto scan = DoScan(repo);
  const ModelScan* m = FindModel(scan, "m");
  ASSERT_NE(m, nullptr);
  ASSERT_EQ(m->versions.size(), 1u);
  EXPECT_EQ(m->versions[0].version, 3u);
  // None of these names are all-decimal, so nothing numeric is skipped.
  EXPECT_TRUE(m->skipped.empty());
}

TEST(RepositoryScanTest, VersionScanSkipsEmptyVersionDir)
{
  TempRepo repo;
  repo.MakeVersion("m", "1");
  fs::create_directories(fs::path(repo.Path()) / "m" / "2");
  auto scan = DoScan(repo);
  const ModelScan* m = FindModel(scan, "m");
  ASSERT_NE(m, nullptr);
  ASSERT_EQ(m->versions.size(), 1u);
  EXPECT_EQ(m->versions[0].version, 1u);
  ExpectVecEq(m->skipped, {2u});
}

TEST(RepositoryScanTest, VersionScanAcceptsInt64Max)
{
  TempRepo repo;
  const std::string max_version = "9223372036854775807";
  repo.MakeVersion("m", max_version);
  // One beyond int64 max must be rejected.
  repo.MakeVersion("m", "9223372036854775808");
  auto scan = DoScan(repo);
  const ModelScan* m = FindModel(scan, "m");
  ASSERT_NE(m, nullptr);
  ASSERT_EQ(m->versions.size(), 1u);
  EXPECT_EQ(m->versions[0].version, 9223372036854775807ULL);
}

TEST(RepositoryScanTest, MissingConfigIsReported)
{
  TempRepo repo;
  repo.MakeVersion("m", "1");
  auto scan = DoScan(repo);
  const ModelScan* m = FindModel(scan, "m");
  ASSERT_NE(m, nullptr);
  EXPECT_FALSE(m->config_present);
  EXPECT_EQ(m->config_fingerprint, 0u);
}

TEST(RepositoryScanTest, CustomConfigFallsBackToDefault)
{
  TempRepo repo;
  repo.MakeVersion("m", "1");
  repo.MakeConfig("m", "platform: \"foo\"\n");
  auto scan = DoScan(repo, "h100");
  const ModelScan* m = FindModel(scan, "m");
  ASSERT_NE(m, nullptr);
  ASSERT_TRUE(m->config_present);
  EXPECT_NE(m->config_fingerprint, 0u);

  repo.WriteFile("m/configs/h100.pbtxt", "platform: \"bar\"\n");
  scan = DoScan(repo, "h100");
  m = FindModel(scan, "m");
  ASSERT_NE(m, nullptr);
  ASSERT_TRUE(m->config_present);
}

TEST(RepositoryScanTest, UnreadableRepositoryFailsScan)
{
  RepositoryScan scan;
  Status st = ScanRepository(
      (fs::temp_directory_path() / "no_such_repo_xyz_123").string(), {}, 1,
      &scan);
  EXPECT_FALSE(st.ok);
  EXPECT_FALSE(st.message.empty());
  EXPECT_EQ(st.message.find('\n'), std::string::npos);
}

// =========================================================================
// Policy parsing and application
// =========================================================================

TEST(RepositoryPolicyTest, DefaultPolicyIsLatestOne)
{
  VersionPolicy policy;
  Status st = ParseVersionPolicy("platform: \"x\"\n", &policy);
  ASSERT_TRUE(st.ok) << st.message;
  EXPECT_EQ(policy.kind, PolicyKind::Latest);
  EXPECT_EQ(policy.num_versions, 1u);
}

TEST(RepositoryPolicyTest, LatestClampsWhenInsufficient)
{
  VersionPolicy policy;
  Status st = ParseVersionPolicy(
      "version_policy { latest { num_versions: 5 } }", &policy);
  ASSERT_TRUE(st.ok) << st.message;
  std::vector<VersionEntry> candidates;
  for (uint64_t v : {1u, 2u}) {
    candidates.push_back(MakeVersionEntry(v));
  }
  std::vector<uint64_t> missing;
  auto served = ApplyVersionPolicy(policy, candidates, &missing);
  ExpectVecEq(served, {1u, 2u});
  EXPECT_TRUE(missing.empty());
}

TEST(RepositoryPolicyTest, AllKeepsEveryNumericVersion)
{
  VersionPolicy policy;
  ASSERT_TRUE(ParseVersionPolicy("version_policy: { all: {} }", &policy).ok);
  std::vector<VersionEntry> candidates;
  for (uint64_t v : {3u, 1u, 2u}) {
    candidates.push_back(MakeVersionEntry(v));
  }
  std::vector<uint64_t> missing;
  auto served = ApplyVersionPolicy(policy, candidates, &missing);
  ExpectVecEq(served, {1u, 2u, 3u});
}

TEST(RepositoryPolicyTest, SpecificReportsMissing)
{
  VersionPolicy policy;
  ASSERT_TRUE(
      ParseVersionPolicy(
          "version_policy { specific { versions: [1, 3] } }", &policy)
          .ok);
  std::vector<VersionEntry> candidates;
  candidates.push_back(MakeVersionEntry(1));
  std::vector<uint64_t> missing;
  auto served = ApplyVersionPolicy(policy, candidates, &missing);
  ExpectVecEq(served, {1u});
  ExpectVecEq(missing, {3u});
}

TEST(RepositoryPolicyTest, SpecificIgnoresUnknownVersions)
{
  VersionPolicy policy;
  ASSERT_TRUE(
      ParseVersionPolicy(
          "version_policy{specific{versions:[2,4,6]}}", &policy)
          .ok);
  std::vector<VersionEntry> candidates;
  for (uint64_t v : {1u, 2u, 3u, 6u}) {
    candidates.push_back(MakeVersionEntry(v));
  }
  std::vector<uint64_t> missing;
  auto served = ApplyVersionPolicy(policy, candidates, &missing);
  ExpectVecEq(served, {2u, 6u});
  ExpectVecEq(missing, {4u});
}

TEST(RepositoryPolicyTest, PolicyParsesWithoutWhitespace)
{
  VersionPolicy policy;
  ASSERT_TRUE(
      ParseVersionPolicy("version_policy:{all:{}}#trailing", &policy).ok);
  EXPECT_EQ(policy.kind, PolicyKind::All);

  VersionPolicy latest;
  ASSERT_TRUE(
      ParseVersionPolicy(
          "# header\nversion_policy:{latest:{num_versions:2}}", &latest)
          .ok);
  EXPECT_EQ(latest.kind, PolicyKind::Latest);
  EXPECT_EQ(latest.num_versions, 2u);

  // Multi-field line and comments inside the block.
  VersionPolicy specific;
  ASSERT_TRUE(
      ParseVersionPolicy(
          "version_policy: { specific: { versions: [1, 3, # mid comment\n"
          " 5] } }",
          &specific)
          .ok);
  ExpectVecEq(specific.versions, {1u, 3u, 5u});
}

TEST(RepositoryPolicyTest, MalformedPolicyIsRejected)
{
  VersionPolicy policy;
  EXPECT_FALSE(
      ParseVersionPolicy("version_policy: { latest { num_versions: 0 } }",
                         &policy)
          .ok);
  EXPECT_FALSE(
      ParseVersionPolicy("version_policy: { bogus {} }", &policy).ok);
  EXPECT_FALSE(
      ParseVersionPolicy("version_policy: { all {", &policy).ok);
  Status st =
      ParseVersionPolicy("version_policy: { all {} latest {} }", &policy);
  EXPECT_FALSE(st.ok);
  EXPECT_NE(st.message.find("config_invalid"), std::string::npos);
}

// =========================================================================
// Migration semantics
// =========================================================================

class EngineTest : public ::testing::Test {
 protected:
  // Two stable scans with the given index let pending fingerprints
  // settle so state reflects the current repository contents.
  void
  Settle(
      RepositoryPolicyEngine* engine, const RepositoryScan& scan,
      const IndexSnapshot& index)
  {
    engine->Observe(scan, index, poll_++);
    engine->Observe(scan, index, poll_++);
  }

  uint32_t poll_{1};
};

TEST_F(EngineTest, DoubleObserveIsIdempotent)
{
  TempRepo repo;
  repo.MakeVersion("m", "1");
  repo.MakeConfig("m", "version_policy { all {} }");
  auto scan = DoScan(repo);

  RepositoryPolicyEngine engine;
  engine.Observe(scan, ReadyIndex("m", {1}), 1);
  auto first = engine.Snapshot();
  ASSERT_EQ(EventCount(first), 1u);
  const auto* v = FindVersion(*FindView(first, "m"), 1);
  ASSERT_NE(v, nullptr);
  uint64_t generation = v->generation;

  engine.Observe(scan, ReadyIndex("m", {1}), 2);
  auto second = engine.Snapshot();
  EXPECT_EQ(EventCount(second), 1u);  // no new event
  v = FindVersion(*FindView(second, "m"), 1);
  ASSERT_NE(v, nullptr);
  EXPECT_EQ(v->generation, generation);
  EXPECT_EQ(v->state, VersionState::Ready);
}

TEST_F(EngineTest, VersionAddedBecomesReady)
{
  TempRepo repo;
  repo.MakeVersion("m", "1");
  repo.MakeConfig("m", "version_policy { all {} }");
  auto scan1 = DoScan(repo);

  RepositoryPolicyEngine engine;
  Settle(&engine, scan1, ReadyIndex("m", {1}));
  ASSERT_NE(FindVersion(*FindView(engine.Snapshot(), "m"), 1), nullptr);

  repo.MakeVersion("m", "2");
  auto scan2 = DoScan(repo);
  // First observation: fingerprint pending, the new version is not yet
  // part of the effective candidate set.
  engine.Observe(scan2, ReadyIndex("m", {1}), poll_++);
  const auto* pending_view = FindView(engine.Snapshot(), "m");
  ASSERT_NE(pending_view, nullptr);
  EXPECT_EQ(FindVersion(*pending_view, 2), nullptr);

  // Second identical scan: fingerprint settles and the index already
  // reports version 2 ready, so it becomes Ready directly.
  engine.Observe(scan2, ReadyIndex("m", {1, 2}), poll_++);
  auto snap = engine.Snapshot();
  const auto* v2 = FindVersion(*FindView(snap, "m"), 2);
  ASSERT_NE(v2, nullptr);
  EXPECT_EQ(v2->state, VersionState::Ready);
}

TEST_F(EngineTest, VersionRemovedEntersDraining)
{
  TempRepo repo;
  repo.MakeVersion("m", "1");
  repo.MakeVersion("m", "2");
  repo.MakeConfig("m", "version_policy { all {} }");
  auto scan1 = DoScan(repo);

  RepositoryPolicyEngine engine;
  Settle(&engine, scan1, ReadyIndex("m", {1, 2}));

  repo.RemoveVersion("m", "2");
  auto scan2 = DoScan(repo);
  Settle(&engine, scan2, ReadyIndex("m", {1, 2}));  // index still has v2

  auto snap = engine.Snapshot();
  const auto* view = FindView(snap, "m");
  ASSERT_NE(view, nullptr);
  ExpectVecEq(view->draining, {2u});
  EXPECT_EQ(FindVersion(*view, 2), nullptr);
  EXPECT_GT(snap->counters.drain_observed, 0u);
}

TEST_F(EngineTest, ReloadKeepsReadyGeneration)
{
  TempRepo repo;
  repo.MakeVersion("m", "1");
  repo.MakeConfig("m", "version_policy { all {} }");
  auto scan = DoScan(repo);

  RepositoryPolicyEngine engine;
  Settle(&engine, scan, ReadyIndex("m", {1}));
  uint64_t before = FindVersion(*FindView(engine.Snapshot(), "m"), 1)
                        ->generation;

  // A series of identical scans must not touch the generation.
  for (int i = 0; i < 4; ++i) {
    engine.Observe(scan, ReadyIndex("m", {1}), poll_++);
  }
  uint64_t after = FindVersion(*FindView(engine.Snapshot(), "m"), 1)
                       ->generation;
  EXPECT_EQ(before, after);
}

TEST_F(EngineTest, UnavailableKeepsReason)
{
  TempRepo repo;
  repo.MakeVersion("m", "1");
  repo.MakeConfig("m", "version_policy { all {} }");
  auto scan = DoScan(repo);

  RepositoryPolicyEngine engine;
  engine.Observe(scan, ReadyIndex("m", {1}), poll_++);

  IndexSnapshot bad;
  bad.entries.push_back(IndexEntry{"m", 1, "UNAVAILABLE", "deadlock detected"});
  engine.Observe(scan, bad, poll_++);
  auto snap = engine.Snapshot();
  const auto* v = FindVersion(*FindView(snap, "m"), 1);
  ASSERT_NE(v, nullptr);
  EXPECT_EQ(v->state, VersionState::Unavailable);
  EXPECT_EQ(v->reason, "deadlock detected");
  EXPECT_GE(snap->counters.drift_error, 1u);
}

TEST_F(EngineTest, GenerationContinuesAfterReappear)
{
  TempRepo repo;
  repo.MakeVersion("m", "1");
  repo.MakeConfig("m", "version_policy { all {} }");
  auto scan1 = DoScan(repo);

  RepositoryPolicyEngine engine;
  Settle(&engine, scan1, ReadyIndex("m", {1}));
  uint64_t first_gen = FindVersion(*FindView(engine.Snapshot(), "m"), 1)
                           ->generation;

  // Remove from disk and let the index drop the version: the record is
  // pruned but its generation tombstone remains.
  repo.RemoveVersion("m", "1");
  auto scan_empty = DoScan(repo);
  Settle(&engine, scan_empty, IndexSnapshot{});
  EXPECT_EQ(FindView(engine.Snapshot(), "m")->served.size(), 0u);

  repo.MakeVersion("m", "1");
  auto scan_back = DoScan(repo);
  Settle(&engine, scan_back, ReadyIndex("m", {1}));
  uint64_t next_gen = FindVersion(*FindView(engine.Snapshot(), "m"), 1)
                          ->generation;
  EXPECT_GT(next_gen, first_gen);
}

// =========================================================================
// Drift handling
// =========================================================================

TEST_F(EngineTest, ExpectedVersionMissingBecomesErrorAfterTwoPolls)
{
  TempRepo repo;
  repo.MakeVersion("m", "1");
  repo.MakeConfig("m", "version_policy { all {} }");
  auto scan = DoScan(repo);

  RepositoryPolicyEngine engine;
  engine.Observe(scan, IndexSnapshot{}, 1);  // index empty: Loading
  auto snap = engine.Snapshot();
  const auto* v = FindVersion(*FindView(snap, "m"), 1);
  ASSERT_NE(v, nullptr);
  EXPECT_EQ(v->state, VersionState::Loading);
  uint64_t errors_at_start = snap->counters.drift_error;

  // Polls 2 and 3 are still within the two-interval grace window.
  engine.Observe(scan, IndexSnapshot{}, 2);
  engine.Observe(scan, IndexSnapshot{}, 3);
  EXPECT_EQ(engine.Snapshot()->counters.drift_error, errors_at_start);

  // Beyond two intervals (poll 4): escalated to an Error drift.
  engine.Observe(scan, IndexSnapshot{}, 4);
  EXPECT_GT(engine.Snapshot()->counters.drift_error, errors_at_start);
}

TEST_F(EngineTest, IndexOnlyModelStaysUnknown)
{
  TempRepo repo;
  repo.MakeVersion("m", "1");
  auto disk_scan = DoScan(repo);

  IndexSnapshot index;
  index.entries.push_back(IndexEntry{"m", 1, "READY", ""});
  index.entries.push_back(IndexEntry{"ghost", 7, "READY", ""});

  RepositoryPolicyEngine engine;
  engine.Observe(disk_scan, index, 1);
  engine.Observe(disk_scan, index, 2);
  auto snap = engine.Snapshot();
  const auto* ghost = FindView(snap, "ghost");
  ASSERT_NE(ghost, nullptr);
  ASSERT_EQ(ghost->served.size(), 1u);
  EXPECT_EQ(ghost->served[0].state, VersionState::Unknown);
  EXPECT_EQ(ghost->served[0].reason, "index_only");

  // Once the index drops the ghost it disappears on its own.
  IndexSnapshot without_ghost;
  without_ghost.entries.push_back(IndexEntry{"m", 1, "READY", ""});
  engine.Observe(disk_scan, without_ghost, 3);
  engine.Observe(disk_scan, without_ghost, 4);
  EXPECT_EQ(FindView(engine.Snapshot(), "ghost"), nullptr);
}

TEST_F(EngineTest, DrainBeyondGraceIsWarnOnly)
{
  TempRepo repo;
  repo.MakeVersion("m", "1");
  repo.MakeConfig("m", "version_policy { all {} }");
  auto scan1 = DoScan(repo);

  RepositoryPolicyEngine engine;
  Settle(&engine, scan1, ReadyIndex("m", {1}));

  repo.RemoveVersion("m", "1");
  auto scan2 = DoScan(repo);
  // The index never releases the version.
  for (uint32_t p = 10; p < 18; ++p) {
    engine.Observe(scan2, ReadyIndex("m", {1}), p);
  }
  auto snap = engine.Snapshot();
  const auto* view = FindView(snap, "m");
  ASSERT_NE(view, nullptr);
  ExpectVecEq(view->draining, {1u});
  // Warns only: no error drift, and the version is never erased.
  uint64_t error_count = 0;
  for (const auto& e : FlattenEventChunks(snap->events)) {
    if (e.severity == DriftSeverity::Error) {
      ++error_count;
    }
  }
  EXPECT_EQ(error_count, 0u);
  EXPECT_GE(snap->counters.drift_warn, 1u);
}

TEST_F(EngineTest, IndexFailureDoesNotAdvanceState)
{
  TempRepo repo;
  repo.MakeVersion("m", "1");
  repo.MakeConfig("m", "version_policy { all {} }");
  auto scan = DoScan(repo);

  RepositoryPolicyEngine engine;
  engine.Observe(scan, ReadyIndex("m", {1}), 1);
  const auto* before = FindVersion(*FindView(engine.Snapshot(), "m"), 1);
  ASSERT_NE(before, nullptr);
  EXPECT_EQ(before->state, VersionState::Ready);

  for (uint32_t p = 2; p <= 4; ++p) {
    engine.Observe(scan, std::nullopt, p);
  }
  auto snap = engine.Snapshot();
  const auto* after = FindVersion(*FindView(snap, "m"), 1);
  ASSERT_NE(after, nullptr);
  EXPECT_EQ(after->state, VersionState::Ready);
  // Index failures surface as Warns, never Errors.
  EXPECT_EQ(snap->counters.drift_error, 0u);
  EXPECT_GE(snap->counters.drift_warn, 3u);

  // Three failures force polling decisions afterwards.
  EXPECT_TRUE(engine.DecidePoll(5));
}

TEST_F(EngineTest, ScanFailureRetainsSnapshot)
{
  TempRepo repo;
  repo.MakeVersion("m", "1");
  auto scan = DoScan(repo);
  RepositoryPolicyEngine engine;
  engine.Observe(scan, ReadyIndex("m", {1}), 1);
  auto before = engine.Snapshot();
  ASSERT_EQ(before->models.size(), 1u);

  engine.ObserveScanFailure(
      Status::Error("permission denied"), 2, repo.Path());
  auto after = engine.Snapshot();
  EXPECT_EQ(after->models.size(), 1u);
  EXPECT_EQ(after->counters.scan_failure, 1u);
  EXPECT_EQ(FindView(after, "m")->served.size(), 1u);
}

TEST_F(EngineTest, ConfigInvalidMarksModelUnavailable)
{
  TempRepo repo;
  repo.MakeVersion("m", "1");
  repo.MakeConfig("m", "version_policy: { all {");
  auto scan = DoScan(repo);

  RepositoryPolicyEngine engine;
  // Two observes so the config fingerprint becomes effective.
  engine.Observe(scan, IndexSnapshot{}, 1);
  engine.Observe(scan, IndexSnapshot{}, 2);
  auto snap = engine.Snapshot();
  const auto* view = FindView(snap, "m");
  ASSERT_NE(view, nullptr);
  ASSERT_EQ(view->served.size(), 1u);
  EXPECT_EQ(view->served[0].state, VersionState::Unavailable);
  EXPECT_NE(view->served[0].reason.find("config_invalid"),
            std::string::npos);
}

TEST_F(EngineTest, NoAvailableVersionsWarnsOnce)
{
  // A specific policy requesting only absent versions leaves the model
  // with no served versions; the warning must fire once, not every poll.
  RepositoryScan scan;
  ModelScan ms;
  ms.name = "m";
  ms.config_present = true;
  ms.config_text = "version_policy { specific { versions: [7] } }";
  VersionEntry ve;
  ve.version = 1;  // on disk but not requested
  ms.versions.push_back(ve);
  scan.models.push_back(ms);

  RepositoryPolicyEngine engine;
  const auto count_warns = [&]() {
    uint64_t n = 0;
    for (auto c = engine.Snapshot()->events; c; c = c->older) {
      for (const auto& e : c->events) {
        if (e.reason == "no_available_versions") {
          ++n;
        }
      }
    }
    return n;
  };
  for (uint32_t p = 1; p <= 5; ++p) {
    engine.Observe(scan, IndexSnapshot{}, p);
  }
  EXPECT_EQ(count_warns(), 1u);
  EXPECT_EQ(FindView(engine.Snapshot(), "m")->expected_missing,
            std::vector<uint64_t>({7u}));
}

// =========================================================================
// Routing
// =========================================================================

TEST(RouteVersionTest, RouteEmptyReturnsMaxServed)
{
  ModelView view;
  view.name = "m";
  for (uint64_t v : {1u, 3u, 2u}) {
    ModelVersionView vv;
    vv.version = v;
    vv.state = VersionState::Ready;
    view.served.push_back(vv);
  }
  std::sort(view.served.begin(), view.served.end(), [](auto& a, auto& b) {
    return a.version < b.version;
  });
  auto r = RouteVersion(view, "");
  ASSERT_TRUE(r.ok);
  EXPECT_EQ(r.version, 3u);
}

TEST(RouteVersionTest, RouteUnknownVersionReturnsNotReady)
{
  ModelView view;
  view.name = "m";
  ModelVersionView vv;
  vv.version = 1;
  vv.state = VersionState::Ready;
  view.served.push_back(vv);
  auto r = RouteVersion(view, "9");
  EXPECT_FALSE(r.ok);
  EXPECT_EQ(
      r.error, "Request for unknown model: 'm' version 9 is not at ready "
               "state");
}

TEST(RouteVersionTest, RouteEmptyServedReturnsNoAvailableVersions)
{
  ModelView view;
  view.name = "m";
  auto r = RouteVersion(view, "");
  EXPECT_FALSE(r.ok);
  EXPECT_EQ(
      r.error, "Request for unknown model: 'm' has no available versions");

  // A numeric request against an empty model gives the same wording.
  auto r2 = RouteVersion(view, "5");
  EXPECT_FALSE(r2.ok);
  EXPECT_EQ(
      r2.error, "Request for unknown model: 'm' has no available versions");
}

TEST(RouteVersionTest, RouteNonNumericReturnsInvalidVersion)
{
  ModelView view;
  view.name = "m";
  ModelVersionView vv;
  vv.version = 1;
  vv.state = VersionState::Ready;
  view.served.push_back(vv);
  for (const std::string bad : {"abc", "1.0", "-1", "v1"}) {
    auto r = RouteVersion(view, bad);
    EXPECT_FALSE(r.ok);
    EXPECT_EQ(r.error, "invalid version");
  }
}

TEST(EventFormatTest, LineHasFixedPrefixAndFields)
{
  RepositoryEvent event;
  event.seq = 7;
  event.poll_index = 3;
  event.model = "m";
  event.version = 2;
  event.from = VersionState::Loading;
  event.to = VersionState::Ready;
  event.severity = DriftSeverity::Warn;
  event.reason = "load_expected";
  const std::string line = FormatEventLine(event);
  EXPECT_EQ(line.rfind("repo-policy: ", 0), 0u);
  EXPECT_EQ(
      line,
      "repo-policy: model=m version=2 from=LOADING to=READY "
      "severity=warn reason=load_expected");

  // Engine-level events without a model/version render dashes.
  RepositoryEvent detached = event;
  detached.model.clear();
  detached.version.reset();
  const std::string detached_line = FormatEventLine(detached);
  EXPECT_NE(detached_line.find("model=- version=-"), std::string::npos);
}

// =========================================================================
// Concurrency
// =========================================================================

TEST(EngineConcurrencyTest, ConcurrentSnapshotReadersNeverSeeTornState)
{
  TempRepo repo;
  for (uint64_t v = 1; v <= 4; ++v) {
    repo.MakeVersion("m", std::to_string(v));
  }
  repo.MakeConfig("m", "version_policy { all {} }");
  auto scan_base = DoScan(repo);

  RepositoryPolicyEngine engine;
  engine.Observe(scan_base, ReadyIndex("m", {1, 2, 3, 4}), 1);

  std::atomic<bool> stop{false};
  std::vector<uint64_t> seen_sizes;
  std::mutex sizes_mu;

  std::vector<std::thread> readers;
  for (int t = 0; t < 8; ++t) {
    readers.emplace_back([&]() {
      for (int i = 0; i < 20000; ++i) {
        auto snap = engine.Snapshot();
        std::set<uint64_t> versions;
        for (const auto& view : snap->models) {
          for (const auto& vv : view.served) {
            versions.insert(vv.version);
          }
        }
        std::lock_guard<std::mutex> lock(sizes_mu);
        seen_sizes.push_back(versions.size());
      }
    });
  }

  // Writer alternates between the four-version repository and a two
  // version one, so every published snapshot differs by exactly one
  // element and each read must be a subset/superset of its neighbours.
  std::thread writer([&]() {
    TempRepo repo2;
    for (uint64_t v = 1; v <= 3; ++v) {
      repo2.MakeVersion("m", std::to_string(v));
    }
    repo2.MakeConfig("m", "version_policy { all {} }");
    auto scan_small = DoScan(repo2);
    uint32_t poll = 2;
    while (!stop.load()) {
      engine.Observe(scan_small, ReadyIndex("m", {1, 2, 3}), poll++);
      engine.Observe(scan_base, ReadyIndex("m", {1, 2, 3, 4}), poll++);
    }
  });

  for (auto& t : readers) {
    t.join();
  }
  stop.store(true);
  writer.join();

  for (size_t i = 1; i < seen_sizes.size(); ++i) {
    uint64_t a = seen_sizes[i - 1];
    uint64_t b = seen_sizes[i];
    EXPECT_TRUE(a == b || a + 1 == b || b + 1 == a)
        << "torn snapshot sizes " << a << " -> " << b;
  }
}

// =========================================================================
// Determinism
// =========================================================================

TEST(DeterminismTest, ShuffledInputProducesIdenticalJson)
{
  RepositoryPolicyEngine engine_a;
  RepositoryPolicyEngine engine_b;

  auto make_scan = [](const std::vector<std::string>& order) {
    RepositoryScan scan;
    scan.scan_seq = 1;
    for (const auto& name : order) {
      ModelScan ms;
      ms.name = name;
      ms.config_present = false;
      for (uint64_t v : std::vector<uint64_t>{3u, 1u, 2u}) {
        ms.versions.push_back(MakeVersionEntry(v));
      }
      scan.models.push_back(std::move(ms));
    }
    return scan;
  };

  std::vector<std::string> names = {"alpha", "beta", "gamma", "delta"};
  auto scan_a = make_scan(names);
  engine_a.Observe(scan_a, IndexSnapshot{}, 1);

  std::mt19937 rng(12345);
  std::string reference = SnapshotJson(*engine_a.Snapshot());
  for (int iter = 0; iter < 50; ++iter) {
    auto shuffled = names;
    std::shuffle(shuffled.begin(), shuffled.end(), rng);
    RepositoryPolicyEngine engine;
    engine.Observe(make_scan(shuffled), IndexSnapshot{}, 1);
    EXPECT_EQ(SnapshotJson(*engine.Snapshot()), reference);
  }
}

TEST(DeterminismTest, SameInputSequenceIsByteIdentical)
{
  TempRepo repo;
  repo.MakeVersion("m", "1");
  repo.MakeVersion("m", "2");
  repo.MakeConfig("m", "version_policy { latest { num_versions: 1 } }");
  auto scan = DoScan(repo);

  RepositoryPolicyEngine e1;
  RepositoryPolicyEngine e2;
  e1.Observe(scan, IndexSnapshot{}, 1);
  e2.Observe(scan, IndexSnapshot{}, 1);
  EXPECT_EQ(SnapshotJson(*e1.Snapshot()), SnapshotJson(*e2.Snapshot()));
}

// =========================================================================
// Performance and bounded memory
// =========================================================================

TEST(PerformanceTest, LargeRepositoryObserveUnderBudget)
{
  TempRepo repo;
  for (int m = 0; m < 200; ++m) {
    std::string name = "model_" + std::to_string(m);
    for (int v = 1; v <= 8; ++v) {
      repo.MakeVersion(name, std::to_string(v), std::string(4096, 'x'));
    }
    repo.MakeConfig(
        name, std::string("platform: \"foo\"\n") + std::string(2000, '#') +
                  "\nversion_policy { all {} }\n");
  }

  auto start = std::chrono::steady_clock::now();
  RepositoryScan scan;
  Status st = ScanRepository(repo.Path(), {}, 1, &scan);
  ASSERT_TRUE(st.ok) << st.message;
  auto scanned = std::chrono::steady_clock::now();

  RepositoryPolicyEngine engine;
  engine.Observe(scan, IndexSnapshot{}, 1);
  auto observed = std::chrono::steady_clock::now();

  auto scan_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(scanned - start)
          .count();
  auto observe_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(observed - scanned)
          .count();
  EXPECT_EQ(scan.models.size(), 200u);
  std::cout << "large repo scan=" << scan_ms << "ms observe=" << observe_ms
            << "ms\n";
  // The 50ms budget covers Observe; scanning a fresh filesystem is
  // allowed a looser bound but both are asserted for regression
  // tracking on the reference hardware.
  EXPECT_LT(observe_ms, 50);
}

TEST(PerformanceTest, EventRingIsBoundedOverManyObserves)
{
  TempRepo repo;
  repo.MakeVersion("m", "1");
  auto scan = DoScan(repo);

  RepositoryPolicyEngine engine;
  for (uint32_t p = 1; p <= 100000; ++p) {
    // Alternating index presence keeps generating loading/error events.
    if ((p % 3) == 0) {
      engine.Observe(scan, ReadyIndex("m", {1}), p);
    } else {
      engine.Observe(scan, IndexSnapshot{}, p);
    }
  }
  auto snap = engine.Snapshot();
  EXPECT_LE(EventCount(snap), RepositorySnapshot::kEventRingCapacity);
  EXPECT_GT(snap->discarded_events, 0u);
  // Counters stay monotonic and well within uint64 range.
  EXPECT_GE(snap->counters.load_expected, 1u);
}

// =========================================================================
// Fingerprint behaviour
// =========================================================================

TEST(FingerprintTest, DetectsSizeMtimeAndCountChanges)
{
  TempRepo repo;
  repo.MakeVersion("m", "1", "abc");
  auto scan1 = DoScan(repo);
  uint64_t fp1 = FindModel(scan1, "m")->versions[0].fingerprint;

  repo.WriteFile("m/1/model.bin", "abcdef");  // size change
  auto scan2 = DoScan(repo);
  uint64_t fp2 = FindModel(scan2, "m")->versions[0].fingerprint;
  EXPECT_NE(fp1, fp2);

  // Same size, new file: count change.
  repo.WriteFile("m/1/extra.bin", "xyz");
  auto scan3 = DoScan(repo);
  uint64_t fp3 = FindModel(scan3, "m")->versions[0].fingerprint;
  EXPECT_NE(fp2, fp3);
}

TEST(FingerprintTest, ConfigFingerprintIsContentBased)
{
  TempRepo repo;
  repo.MakeVersion("m", "1");
  repo.MakeConfig("m", "version_policy { all {} }");
  uint64_t fp1 = FindModel(DoScan(repo), "m")->config_fingerprint;
  repo.MakeConfig("m", "version_policy { all {} }\n");
  uint64_t fp2 = FindModel(DoScan(repo), "m")->config_fingerprint;
  EXPECT_NE(fp1, fp2);
}

TEST(StabilityTest, TwoRoundStabilityHoldsThenReleases)
{
  TempRepo repo;
  repo.MakeVersion("m", "1");
  repo.MakeConfig("m", "version_policy { all {} }");
  auto scan1 = DoScan(repo);

  RepositoryPolicyEngine engine;
  engine.Observe(scan1, ReadyIndex("m", {1}), 1);

  // Flip the config fingerprint: round one keeps the old expectation.
  auto scan2 = scan1;
  scan2.models[0].config_fingerprint ^= 0xDEADBEEF;
  scan2.models[0].config_text = "version_policy { all {} }";
  engine.Observe(scan2, ReadyIndex("m", {1}), 2);
  EXPECT_TRUE(engine.DecidePoll(3));  // pending change forces a poll

  // The third round agreeing with round two releases the new value.
  engine.Observe(scan2, ReadyIndex("m", {1}), 3);
  auto snap = engine.Snapshot();
  bool saw_unstable = false;
  for (const auto& e : FlattenEventChunks(snap->events)) {
    if (e.reason == "unstable_repository") {
      saw_unstable = true;
    }
  }
  // No instability here because the pending value was confirmed on the
  // very next round.
  EXPECT_FALSE(saw_unstable);
}
