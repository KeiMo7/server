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
// Offline repository policy checker. Runs the same scan, policy and
// reconciliation logic the server uses in its poll loop, but without a
// running Triton server: the index input is always absent so the tool
// reports disk-side expectations and configuration problems.
//
// Single-shot mode prints deterministic JSON and returns 0, or 2 when
// an Error-level drift is found. --watch re-scans on
// --repository-poll-secs boundaries and prints each new event as a
// "repo-policy:" prefixed line.
//
// The argument parser is intentionally local and minimal; the server's
// command_line_parser must not be reused here.

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "repository_policy.h"
#include "repository_policy_scan.h"

namespace triton { namespace server { namespace repository {
namespace {

std::atomic<bool> g_watch_exit{false};

void
HandleSignal(int)
{
  g_watch_exit.store(true);
}

struct Options {
  std::vector<std::string> repositories;
  std::string model_config_name;
  uint32_t repository_poll_secs{15};
  bool watch{false};
  bool json{false};
  bool ok{true};
  std::string error;
};

Options
ParseArgs(int argc, char** argv)
{
  Options opts;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    auto need_value = [&](const std::string& flag) -> std::string {
      std::string prefix = "--" + flag;
      if (arg.rfind(prefix + "=", 0) == 0) {
        return arg.substr(prefix.size() + 1);
      }
      if (arg == prefix) {
        if (i + 1 < argc) {
          return std::string(argv[++i]);
        }
        opts.ok = false;
        opts.error = "missing value for " + prefix;
        return std::string();
      }
      opts.ok = false;
      opts.error = "unrecognized argument: " + arg;
      return std::string();
    };

    if (arg.rfind("--model-repository=", 0) == 0 ||
        arg == "--model-repository") {
      std::string value = need_value("model-repository");
      if (!opts.ok) {
        return opts;
      }
      opts.repositories.push_back(value);
    } else if (arg.rfind("--model-config-name=", 0) == 0 ||
               arg == "--model-config-name") {
      opts.model_config_name = need_value("model-config-name");
      if (!opts.ok) {
        return opts;
      }
    } else if (arg.rfind("--repository-poll-secs=", 0) == 0 ||
               arg == "--repository-poll-secs") {
      std::string value = need_value("repository-poll-secs");
      if (!opts.ok) {
        return opts;
      }
      try {
        int parsed = std::stoi(value);
        if (parsed <= 0) {
          opts.ok = false;
          opts.error = "--repository-poll-secs must be positive";
          return opts;
        }
        opts.repository_poll_secs = static_cast<uint32_t>(parsed);
      }
      catch (...) {
        opts.ok = false;
        opts.error = "invalid --repository-poll-secs value: " + value;
        return opts;
      }
    } else if (arg == "--watch") {
      opts.watch = true;
    } else if (arg == "--json") {
      opts.json = true;
    } else if (arg == "--help" || arg == "-h") {
      opts.ok = false;
      opts.error = "help";
      return opts;
    } else {
      opts.ok = false;
      opts.error = "unrecognized argument: " + arg;
      return opts;
    }
  }

  if (opts.repositories.empty()) {
    opts.ok = false;
    opts.error = "at least one --model-repository is required";
  }
  return opts;
}

nlohmann::ordered_json
SnapshotToJson(const RepositorySnapshot& snap)
{
  nlohmann::ordered_json root;
  root["seq"] = snap.seq;
  root["poll_index"] = snap.poll_index;

  nlohmann::ordered_json models = nlohmann::ordered_json::array();
  for (const ModelView& view : snap.models) {
    nlohmann::ordered_json mj;
    mj["name"] = view.name;
    mj["policy"] = PolicyKindName(view.policy.kind);
    if (view.policy.kind == PolicyKind::Latest) {
      mj["num_versions"] = view.policy.num_versions;
    }
    if (view.policy.kind == PolicyKind::Specific) {
      mj["versions"] = view.policy.versions;
    }

    nlohmann::ordered_json served = nlohmann::ordered_json::array();
    for (const ModelVersionView& vv : view.served) {
      nlohmann::ordered_json vj;
      vj["version"] = vv.version;
      vj["state"] = IndexStateName(vv.state);
      vj["reason"] = vv.reason;
      vj["generation"] = vv.generation;
      served.push_back(std::move(vj));
    }
    mj["served"] = std::move(served);
    mj["expected_missing"] = view.expected_missing;
    mj["draining"] = view.draining;
    models.push_back(std::move(mj));
  }
  root["models"] = std::move(models);

  nlohmann::ordered_json counters;
  counters["load_expected"] = snap.counters.load_expected;
  counters["unload_expected"] = snap.counters.unload_expected;
  counters["drift_error"] = snap.counters.drift_error;
  counters["drift_warn"] = snap.counters.drift_warn;
  counters["drain_observed"] = snap.counters.drain_observed;
  counters["scan_failure"] = snap.counters.scan_failure;
  counters["skipped_polls"] = snap.counters.skipped_polls;
  root["counters"] = std::move(counters);
  return root;
}

bool
HasErrorDrift(const RepositorySnapshot& snap)
{
  for (const ModelView& view : snap.models) {
    for (const ModelVersionView& vv : view.served) {
      if (vv.state == VersionState::Unavailable) {
        return true;
      }
    }
    if (!view.expected_missing.empty()) {
      return true;
    }
  }
  for (const RepositoryEvent& event : FlattenEventChunks(snap.events)) {
    if (event.severity == DriftSeverity::Error) {
      return true;
    }
  }
  return false;
}

void
PrintUsage()
{
  std::cerr
      << "Usage: repository_policy_check --model-repository=<path> "
         "[--model-repository=<path> ...]\n"
         "                              [--model-config-name=<name>]\n"
         "                              [--repository-poll-secs=<secs>]\n"
         "                              [--watch] [--json]\n";
}

}  // namespace
}}}  // namespace triton::server::repository

int
main(int argc, char** argv)
{
  using namespace triton::server::repository;

  Options opts = ParseArgs(argc, argv);
  if (!opts.ok) {
    if (opts.error != "help") {
      std::cerr << "error: " << opts.error << std::endl;
    }
    PrintUsage();
    return opts.error == "help" ? 0 : 1;
  }

  for (const std::string& root : opts.repositories) {
    if (IsCloudRepositoryPath(root)) {
      std::cout << "repo-policy: model=- version=- from=UNKNOWN to=UNKNOWN "
                   "severity=warn reason=cloud_repository_skipped path="
                << root << std::endl;
    }
  }

  RepositoryPolicyEngine engine(opts.model_config_name);
  uint64_t scan_seq = 0;

  auto run_scan = [&](uint32_t poll_index) -> bool {
    RepositoryScan scan;
    Status st = ScanRepositories(
        opts.repositories, opts.model_config_name, ++scan_seq, &scan);
    if (!st.ok) {
      engine.ObserveScanFailure(st, poll_index, opts.repositories.front());
      return false;
    }
    // No live server index is available offline.
    engine.Observe(scan, std::nullopt, poll_index);
    return true;
  };

  if (!opts.watch) {
    bool scan_ok = run_scan(0);
    auto snap = engine.Snapshot();
    // Single-shot mode always emits the deterministic JSON document.
    std::cout << SnapshotToJson(*snap).dump(2) << std::endl;
    if (!scan_ok) {
      return 2;
    }
    return HasErrorDrift(*snap) ? 2 : 0;
  }

  std::signal(SIGINT, HandleSignal);
  std::signal(SIGTERM, HandleSignal);

  uint32_t poll_index = 0;
  uint64_t last_event_seq = 0;
  while (!g_watch_exit.load()) {
    run_scan(poll_index);
    for (const RepositoryEvent& event : engine.Events()) {
      if (event.seq > last_event_seq) {
        std::cout << FormatEventLine(event) << std::endl;
      }
      last_event_seq = std::max(last_event_seq, event.seq);
    }
    ++poll_index;
    for (uint32_t waited = 0;
         waited < opts.repository_poll_secs && !g_watch_exit.load();
         ++waited) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
  return 0;
}
