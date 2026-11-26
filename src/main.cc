// Copyright 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#endif

#ifndef _WIN32
#include <getopt.h>
#include <unistd.h>
#endif

#include <stdint.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "triton_signal.h"

#ifdef TRITON_ENABLE_ASAN
#include <sanitizer/lsan_interface.h>
#endif  // TRITON_ENABLE_ASAN

#include "command_line_parser.h"
#include "common.h"
#include "repository_policy.h"
#include "repository_policy_scan.h"
#include "shared_memory_manager.h"
#include "tracer.h"
#include "triton/common/logging.h"
#include "triton/core/tritonserver.h"

#if defined(TRITON_ENABLE_HTTP) || defined(TRITON_ENABLE_METRICS)
#include "http_server.h"
#endif  // TRITON_ENABLE_HTTP|| TRITON_ENABLE_METRICS
#ifdef TRITON_ENABLE_SAGEMAKER
#include "sagemaker_server.h"
#endif  // TRITON_ENABLE_SAGEMAKER
#ifdef TRITON_ENABLE_VERTEX_AI
#include "vertex_ai_server.h"
#endif  // TRITON_ENABLE_VERTEX_AI
#ifdef TRITON_ENABLE_GRPC
#include "grpc/grpc_server.h"
#endif  // TRITON_ENABLE_GRPC

#ifdef TRITON_ENABLE_GPU
static_assert(
    TRITON_MIN_COMPUTE_CAPABILITY >= 1.0,
    "Invalid TRITON_MIN_COMPUTE_CAPABILITY specified");
#endif  // TRITON_ENABLE_GPU

namespace {

#ifdef TRITON_ENABLE_HTTP
std::unique_ptr<triton::server::HTTPServer> g_http_service;
#endif  // TRITON_ENABLE_HTTP

#ifdef TRITON_ENABLE_GRPC
std::unique_ptr<triton::server::grpc::Server> g_grpc_service;
#endif  // TRITON_ENABLE_GRPC

#ifdef TRITON_ENABLE_METRICS
std::unique_ptr<triton::server::HTTPServer> g_metrics_service;
#endif  // TRITON_ENABLE_METRICS

#ifdef TRITON_ENABLE_SAGEMAKER
std::unique_ptr<triton::server::HTTPServer> g_sagemaker_service;
#endif  // TRITON_ENABLE_SAGEMAKER

#ifdef TRITON_ENABLE_VERTEX_AI
std::unique_ptr<triton::server::HTTPServer> g_vertex_ai_service;
#endif  // TRITON_ENABLE_VERTEX_AI

triton::server::TritonServerParameters g_triton_params;

#ifdef TRITON_ENABLE_GRPC
TRITONSERVER_Error*
StartGrpcService(
    std::unique_ptr<triton::server::grpc::Server>* service,
    const std::shared_ptr<TRITONSERVER_Server>& server,
    triton::server::TraceManager* trace_manager,
    const std::shared_ptr<triton::server::SharedMemoryManager>& shm_manager)
{
  TRITONSERVER_Error* err = triton::server::grpc::Server::Create(
      server, trace_manager, shm_manager, g_triton_params.grpc_options_,
      service);
  if (err == nullptr) {
    err = (*service)->Start();
  }

  if (err != nullptr) {
    service->reset();
  }

  return err;
}
#endif  // TRITON_ENABLE_GRPC

#ifdef TRITON_ENABLE_HTTP
TRITONSERVER_Error*
StartHttpService(
    std::unique_ptr<triton::server::HTTPServer>* service,
    const std::shared_ptr<TRITONSERVER_Server>& server,
    triton::server::TraceManager* trace_manager,
    const std::shared_ptr<triton::server::SharedMemoryManager>& shm_manager)
{
  TRITONSERVER_Error* err = triton::server::HTTPAPIServer::Create(
      server, trace_manager, shm_manager, g_triton_params.http_port_,
      g_triton_params.reuse_http_port_, g_triton_params.http_address_,
      g_triton_params.http_forward_header_pattern_,
      g_triton_params.http_thread_cnt_, g_triton_params.http_max_input_size_,
      g_triton_params.http_restricted_apis_, service);
  if (err == nullptr) {
    err = (*service)->Start();
  }

  if (err != nullptr) {
    service->reset();
  }

  return err;
}
#endif  // TRITON_ENABLE_HTTP

#ifdef TRITON_ENABLE_METRICS
TRITONSERVER_Error*
StartMetricsService(
    std::unique_ptr<triton::server::HTTPServer>* service,
    const std::shared_ptr<TRITONSERVER_Server>& server)
{
  TRITONSERVER_Error* err = triton::server::HTTPMetricsServer::Create(
      server, g_triton_params.metrics_port_, g_triton_params.metrics_address_,
      1 /* HTTP thread count */, service);
  if (err == nullptr) {
    err = (*service)->Start();
  }
  if (err != nullptr) {
    service->reset();
  }

  return err;
}
#endif  // TRITON_ENABLE_METRICS

#ifdef TRITON_ENABLE_SAGEMAKER
TRITONSERVER_Error*
StartSagemakerService(
    std::unique_ptr<triton::server::HTTPServer>* service,
    const std::shared_ptr<TRITONSERVER_Server>& server,
    triton::server::TraceManager* trace_manager,
    const std::shared_ptr<triton::server::SharedMemoryManager>& shm_manager)
{
  size_t max_input_size = triton::server::HTTP_DEFAULT_MAX_INPUT_SIZE;
  triton::server::RestrictedFeatures restricted_apis{};
#ifdef TRITON_ENABLE_HTTP
  // Reuse HTTP server settings for SageMaker endpoint behavior. In
  // particular, --http-restricted-api and --http-max-input-size also apply
  // to SageMaker requests. Without TRITON_ENABLE_HTTP, SageMaker falls back to
  // default input size and unrestricted APIs (no command-line configuration for
  // restricted APIs).
  max_input_size = g_triton_params.http_max_input_size_;
  restricted_apis = g_triton_params.http_restricted_apis_;
#endif  // TRITON_ENABLE_HTTP

  TRITONSERVER_Error* err = triton::server::SagemakerAPIServer::Create(
      server, trace_manager, shm_manager, g_triton_params.sagemaker_port_,
      g_triton_params.sagemaker_address_, g_triton_params.sagemaker_thread_cnt_,
      max_input_size, restricted_apis, service);
  if (err == nullptr) {
    err = (*service)->Start();
  }

  if (err != nullptr) {
    service->reset();
  }

  return err;
}
#endif  // TRITON_ENABLE_SAGEMAKER

#ifdef TRITON_ENABLE_VERTEX_AI
TRITONSERVER_Error*
StartVertexAiService(
    std::unique_ptr<triton::server::HTTPServer>* service,
    const std::shared_ptr<TRITONSERVER_Server>& server,
    triton::server::TraceManager* trace_manager,
    const std::shared_ptr<triton::server::SharedMemoryManager>& shm_manager)
{
  size_t max_input_size = triton::server::HTTP_DEFAULT_MAX_INPUT_SIZE;
  triton::server::RestrictedFeatures restricted_apis{};
#ifdef TRITON_ENABLE_HTTP
  // Reuse HTTP server settings for Vertex AI endpoint behavior. In
  // particular, --http-restricted-api and --http-max-input-size also apply
  // to Vertex AI requests. Without TRITON_ENABLE_HTTP, Vertex AI falls back to
  // default input size and unrestricted APIs (no command-line configuration for
  // restricted APIs).
  max_input_size = g_triton_params.http_max_input_size_;
  restricted_apis = g_triton_params.http_restricted_apis_;
#endif  // TRITON_ENABLE_HTTP

  TRITONSERVER_Error* err = triton::server::VertexAiAPIServer::Create(
      server, trace_manager, shm_manager, g_triton_params.vertex_ai_port_,
      g_triton_params.vertex_ai_address_, g_triton_params.vertex_ai_thread_cnt_,
      max_input_size, restricted_apis, g_triton_params.vertex_ai_default_model_,
      service);
  if (err == nullptr) {
    err = (*service)->Start();
  }

  if (err != nullptr) {
    service->reset();
  }

  return err;
}
#endif  // TRITON_ENABLE_VERTEX_AI

bool
StartEndpoints(
    const std::shared_ptr<TRITONSERVER_Server>& server,
    triton::server::TraceManager* trace_manager,
    const std::shared_ptr<triton::server::SharedMemoryManager>& shm_manager)
{
#ifdef _WIN32
  WSADATA wsaData;
  int wsa_ret = WSAStartup(MAKEWORD(2, 2), &wsaData);

  if (wsa_ret != 0) {
    LOG_ERROR << "Error in WSAStartup " << wsa_ret;
    return false;
  }
#endif

#ifdef TRITON_ENABLE_GRPC
  // Enable GRPC endpoints if requested...
  if (g_triton_params.allow_grpc_) {
    TRITONSERVER_Error* err =
        StartGrpcService(&g_grpc_service, server, trace_manager, shm_manager);
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to start GRPC service");
      return false;
    }
  }
#endif  // TRITON_ENABLE_GRPC

#ifdef TRITON_ENABLE_HTTP
  // Enable HTTP endpoints if requested...
  if (g_triton_params.allow_http_) {
    TRITONSERVER_Error* err =
        StartHttpService(&g_http_service, server, trace_manager, shm_manager);
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to start HTTP service");
      return false;
    }
  }
#endif  // TRITON_ENABLE_HTTP


#ifdef TRITON_ENABLE_SAGEMAKER
  // Enable Sagemaker endpoints if requested...
  if (g_triton_params.allow_sagemaker_) {
    TRITONSERVER_Error* err = StartSagemakerService(
        &g_sagemaker_service, server, trace_manager, shm_manager);
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to start Sagemaker service");
      return false;
    }
  }
#endif  // TRITON_ENABLE_SAGEMAKER

#ifdef TRITON_ENABLE_VERTEX_AI
  // Enable Vertex AI endpoints if requested...
  if (g_triton_params.allow_vertex_ai_) {
    TRITONSERVER_Error* err = StartVertexAiService(
        &g_vertex_ai_service, server, trace_manager, shm_manager);
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to start Vertex AI service");
      return false;
    }
  }
#endif  // TRITON_ENABLE_VERTEX_AI

#ifdef TRITON_ENABLE_METRICS
  // Enable metrics endpoint if requested...
  if (g_triton_params.allow_metrics_) {
    TRITONSERVER_Error* err = StartMetricsService(&g_metrics_service, server);
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to start Metrics service");
      return false;
    }
  }
#endif  // TRITON_ENABLE_METRICS

  return true;
}

bool
StopEndpoints(uint32_t* exit_timeout_secs)
{
  bool ret = true;

#ifdef TRITON_ENABLE_HTTP
  if (g_http_service) {
    TRITONSERVER_Error* err = g_http_service->Stop(exit_timeout_secs);
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to stop HTTP service");
      ret = false;
    }

    g_http_service.reset();
  }
#endif  // TRITON_ENABLE_HTTP

#ifdef TRITON_ENABLE_GRPC
  // Allow for graceful shutdown of GRPC service
  if (g_grpc_service) {
    TRITONSERVER_Error* err = g_grpc_service->GracefulStop(exit_timeout_secs);
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to gracefully stop GRPC service");
      ret = false;
    }
  }
#endif  // TRITON_ENABLE_GRPC

  return ret;
}

bool
StopEndpoints()
{
  bool ret = true;

#ifdef TRITON_ENABLE_GRPC
  if (g_grpc_service) {
    // Forceful shutdown of GRPC service
    TRITONSERVER_Error* err = g_grpc_service->Stop();
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to stop GRPC service");
      ret = false;
    }

    g_grpc_service.reset();
  }
#endif  // TRITON_ENABLE_GRPC

#ifdef TRITON_ENABLE_METRICS
  if (g_metrics_service) {
    TRITONSERVER_Error* err = g_metrics_service->Stop();
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to stop Metrics service");
      ret = false;
    }

    g_metrics_service.reset();
  }
#endif  // TRITON_ENABLE_METRICS

#ifdef TRITON_ENABLE_SAGEMAKER
  if (g_sagemaker_service) {
    TRITONSERVER_Error* err = g_sagemaker_service->Stop();
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to stop Sagemaker service");
      ret = false;
    }

    g_sagemaker_service.reset();
  }
#endif  // TRITON_ENABLE_SAGEMAKER

#ifdef TRITON_ENABLE_VERTEX_AI
  if (g_vertex_ai_service) {
    TRITONSERVER_Error* err = g_vertex_ai_service->Stop();
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "failed to stop Vertex AI service");
      ret = false;
    }

    g_vertex_ai_service.reset();
  }
#endif  // TRITON_ENABLE_VERTEX_AI

#ifdef _WIN32
  int wsa_ret = WSACleanup();

  if (wsa_ret != 0) {
    LOG_ERROR << "Error in WSACleanup " << wsa_ret;
    ret = false;
  }
#endif

  return ret;
}

bool
StartTracing(triton::server::TraceManager** trace_manager)
{
  *trace_manager = nullptr;

#ifdef TRITON_ENABLE_TRACING
  TRITONSERVER_Error* err = triton::server::TraceManager::Create(
      trace_manager, g_triton_params.trace_level_, g_triton_params.trace_rate_,
      g_triton_params.trace_count_, g_triton_params.trace_log_frequency_,
      g_triton_params.trace_filepath_, g_triton_params.trace_mode_,
      g_triton_params.trace_config_map_);

  if (err != nullptr) {
    LOG_TRITONSERVER_ERROR(err, "failed to configure tracing");
    if (*trace_manager != nullptr) {
      delete (*trace_manager);
    }
    *trace_manager = nullptr;
    return false;
  }
#endif  // TRITON_ENABLE_TRACING

  return true;
}

bool
StopTracing(triton::server::TraceManager** trace_manager)
{
#ifdef TRITON_ENABLE_TRACING
  // We assume that at this point Triton has been stopped gracefully,
  // so can delete the trace manager to finalize the output.
  delete (*trace_manager);
  *trace_manager = nullptr;
#endif  // TRITON_ENABLE_TRACING

  return true;
}

namespace repo = triton::server::repository;

// Convert the server's /v2/repository/index JSON message into the
// backend-neutral IndexSnapshot the policy module consumes. This is the
// single place in the server that knows about the index wire format.
bool
BuildIndexSnapshot(
    const char* buffer, size_t byte_size, repo::IndexSnapshot* snapshot,
    std::string* error)
{
  try {
    nlohmann::json parsed = nlohmann::json::parse(buffer, buffer + byte_size);
    if (!parsed.is_array()) {
      *error = "repository index JSON is not an array";
      return false;
    }
    for (const auto& row : parsed) {
      repo::IndexEntry entry;
      entry.name = row.value("name", "");
      entry.state = row.value("state", "");
      entry.reason = row.value("reason", "");
      if (row.contains("version") && !row["version"].is_null() &&
          row["version"].is_string()) {
        const std::string version_text = row["version"].get<std::string>();
        if (!version_text.empty()) {
          try {
            entry.version = std::stoull(version_text);
          }
          catch (...) {
            entry.version.reset();
          }
        }
      }
      snapshot->entries.push_back(std::move(entry));
    }
  }
  catch (const std::exception& e) {
    *error = std::string("failed to parse repository index JSON: ") + e.what();
    return false;
  }
  return true;
}

// Fetch the live index through libtritonserver and convert it. Returns
// false (without raising policy drift) on any failure.
bool
FetchIndexSnapshot(TRITONSERVER_Server* server, repo::IndexSnapshot* snapshot)
{
  TRITONSERVER_Message* message = nullptr;
  TRITONSERVER_Error* err =
      TRITONSERVER_ServerModelIndex(server, 0 /* all states */, &message);
  if (err != nullptr) {
    LOG_TRITONSERVER_ERROR(err, "failed to read repository index");
    return false;
  }
  const char* buffer = nullptr;
  size_t byte_size = 0;
  err = TRITONSERVER_MessageSerializeToJson(message, &buffer, &byte_size);
  if (err != nullptr) {
    LOG_TRITONSERVER_ERROR(err, "failed to serialize repository index");
    TRITONSERVER_MessageDelete(message);
    return false;
  }
  std::string parse_error;
  bool ok = BuildIndexSnapshot(buffer, byte_size, snapshot, &parse_error);
  if (!ok) {
    LOG_WARNING << parse_error;
  }
  TRITONSERVER_MessageDelete(message);
  return ok;
}

// Pre-flight repository scan performed before TRITONSERVER_ServerNew.
// Cloud-backed repositories are skipped with a warning; local paths are
// scanned and every model gets one I-level summary line. In strict mode
// any configuration parse failure, empty candidate set or unsatisfied
// specific policy aborts startup with exit(1).
bool
PreflightRepositoryPolicy(
    const std::set<std::string>& repository_paths,
    const std::string& model_config_name, bool strict)
{
  auto policy_engine =
      std::make_unique<repo::RepositoryPolicyEngine>(model_config_name);
  bool strict_failure = false;
  uint64_t scan_seq = 0;
  std::vector<std::string> local_roots;

  for (const auto& path : repository_paths) {
    if (repo::IsCloudRepositoryPath(path)) {
      LOG_WARNING << "repo-policy: model=- version=- from=UNKNOWN to=UNKNOWN "
                  "severity=warn reason=cloud_repository_skipped path="
               << path;
      continue;
    }
    local_roots.push_back(path);
  }

  repo::RepositoryScan scan;
  repo::Status status = repo::ScanRepositories(
      local_roots, model_config_name, ++scan_seq, &scan);
  if (!status.ok) {
    if (strict) {
      LOG_ERROR << "repo-policy: model=- version=- from=UNKNOWN to=UNKNOWN "
                   "severity=error reason="
                << status.message;
      return false;
    }
    LOG_WARNING << "repo-policy: model=- version=- from=UNKNOWN to=UNKNOWN "
                "severity=warn reason="
             << status.message;
    return true;
  }

  // No live index exists before the server is created.
  policy_engine->Observe(scan, std::nullopt, 0);

  for (const repo::ModelScan& model : scan.models) {
    repo::VersionPolicy policy;
    repo::Status parse_status = repo::Status::OK();
    if (model.config_present) {
      parse_status = repo::ParseVersionPolicy(model.config_text, &policy);
    }
    std::vector<uint64_t> missing;
    std::vector<uint64_t> expected =
        repo::ApplyVersionPolicy(policy, model.versions, &missing);

    LOG_INFO << "repo-policy: model=" << model.name
             << " versions_on_disk=" << model.versions.size()
             << " expected_served=" << expected.size()
             << " expected_missing=" << missing.size()
             << " config_present=" << (model.config_present ? 1 : 0)
             << " strict=" << (strict ? 1 : 0);

    if (!parse_status.ok) {
      strict_failure = true;
      LOG_ERROR << "repo-policy: model=" << model.name
                << " version=- from=UNKNOWN to=UNAVAILABLE severity=error "
                   "reason="
                << parse_status.message;
    }
    if (model.versions.empty()) {
      strict_failure = true;
      LOG_ERROR << "repo-policy: model=" << model.name
                << " version=- from=UNKNOWN to=UNAVAILABLE severity=error "
                   "reason=no_available_versions";
    }
    if (!missing.empty()) {
      strict_failure = true;
      std::ostringstream oss;
      oss << "expected_missing";
      for (uint64_t v : missing) {
        oss << ":" << v;
      }
      LOG_ERROR << "repo-policy: model=" << model.name
                << " version=- from=UNKNOWN to=UNAVAILABLE severity=error "
                   "reason="
                << oss.str();
    }
  }

  if (strict && strict_failure) {
    return false;
  }
  return true;
}

}  // namespace

int
main(int argc, char** argv)
{
  // Parse command-line to create the options for the inference
  // server.
  triton::server::TritonParser tp;
  try {
    auto res = tp.Parse(argc, argv);
    g_triton_params = res.first;
    g_triton_params.CheckPortCollision();
  }
  catch (const triton::server::ParseException& pe) {
    std::cerr << "Usage: tritonserver [options]" << std::endl;
    std::cerr << tp.Usage() << std::endl;
    // Show error at bottom for immediate visibility
    std::cerr << pe.what() << std::endl;
    exit(1);
  }

  triton::server::TritonServerParameters::ManagedTritonServerOptionPtr
      triton_options(nullptr, TRITONSERVER_ServerOptionsDelete);
  try {
    triton_options = g_triton_params.BuildTritonServerOptions();
  }
  catch (const triton::server::ParseException& pe) {
    std::cerr << "Failed to build Triton option:" << std::endl;
    std::cerr << pe.what() << std::endl;
    exit(1);
  }

#ifdef TRITON_ENABLE_LOGGING
  // Initialize our own logging instance since it is used by GRPC and
  // HTTP endpoints. This logging instance is separate from the one in
  // libtritonserver so we must initialize explicitly.
  LOG_ENABLE_INFO(g_triton_params.log_info_);
  LOG_ENABLE_WARNING(g_triton_params.log_warn_);
  LOG_ENABLE_ERROR(g_triton_params.log_error_);
  LOG_SET_VERBOSE(g_triton_params.log_verbose_);
  LOG_SET_FORMAT(g_triton_params.log_format_);
  LOG_SET_OUT_FILE(g_triton_params.log_file_);
#endif  // TRITON_ENABLE_LOGGING

  // Trace manager.
  triton::server::TraceManager* trace_manager;

  // Manager for shared memory blocks.
  auto shm_manager = std::make_shared<triton::server::SharedMemoryManager>(
      g_triton_params.allow_client_shm_);

  // Backend-independent pre-flight repository policy check. Runs before
  // the server is created and only reads the local filesystem. In strict
  // mode a policy failure ends the process here; --strict-version-policy
  // takes precedence over --exit-on-error for this check only.
  if (!PreflightRepositoryPolicy(
          g_triton_params.model_repository_paths_,
          g_triton_params.model_config_name_,
          g_triton_params.strict_version_policy_)) {
    LOG_ERROR << "strict version policy pre-flight check failed";
    exit(1);
  }

  // Policy reconciliation engine for the poll loop. It only produces
  // expectations and events; all loading/unloading stays with
  // libtritonserver.
  auto policy_engine = std::make_unique<repo::RepositoryPolicyEngine>(
      g_triton_params.model_config_name_);
  uint64_t policy_scan_seq = 0;
  uint32_t policy_poll_index = 0;
  uint64_t last_logged_event_seq = 0;

  // Create the server...
  TRITONSERVER_Server* server_ptr = nullptr;
  FAIL_IF_ERR(
      TRITONSERVER_ServerNew(&server_ptr, triton_options.get()),
      "creating server");

  std::shared_ptr<TRITONSERVER_Server> server(
      server_ptr, TRITONSERVER_ServerDelete);

  // Configure and start tracing if specified on the command line.
  if (!StartTracing(&trace_manager)) {
    exit(1);
  }

  // Trap SIGINT and SIGTERM to allow server to exit gracefully
  TRITONSERVER_Error* signal_err = triton::server::RegisterSignalHandler();
  if (signal_err != nullptr) {
    LOG_TRITONSERVER_ERROR(signal_err, "failed to register signal handler");
    exit(1);
  }

  // Start the HTTP, GRPC, and metrics endpoints.
  if (!StartEndpoints(server, trace_manager, shm_manager)) {
    exit(1);
  }

  // Wait until a signal terminates the server...
  while (!triton::server::signal_exiting_) {
    // If enabled, poll the model repository to see if there have been
    // any changes. DecidePoll() suppresses redundant polls when nothing
    // on disk changed but always forces one at the interval cap, so a
    // missed change can never persist.
    if (g_triton_params.repository_poll_secs_ > 0) {
      if (policy_engine->DecidePoll(policy_poll_index)) {
        LOG_TRITONSERVER_ERROR(
            TRITONSERVER_ServerPollModelRepository(server_ptr),
            "failed to poll model repository");

        // Re-scan the disk side and reconcile with the fresh index.
        std::vector<std::string> local_roots;
        for (const auto& path : g_triton_params.model_repository_paths_) {
          if (!repo::IsCloudRepositoryPath(path)) {
            local_roots.push_back(path);
          }
        }
        repo::RepositoryScan disk_scan;
        repo::Status scan_status = repo::ScanRepositories(
            local_roots, g_triton_params.model_config_name_,
            ++policy_scan_seq, &disk_scan);
        if (!scan_status.ok) {
          policy_engine->ObserveScanFailure(
              scan_status, policy_poll_index,
              local_roots.empty() ? std::string() : local_roots.front());
        } else {
          repo::IndexSnapshot index_snapshot;
          if (FetchIndexSnapshot(server_ptr, &index_snapshot)) {
            policy_engine->Observe(
                disk_scan, std::move(index_snapshot), policy_poll_index);
          } else {
            // Index failure: refresh disk expectations only; the engine
            // records a Warn and never advances version state.
            policy_engine->Observe(
                disk_scan, std::nullopt, policy_poll_index);
          }
        }

        // Emit only the events produced since the previous poll; the
        // engine's ring retains history but the log must not repeat it.
        for (const auto& event : policy_engine->Events()) {
          if (event.seq <= last_logged_event_seq) {
            continue;
          }
          last_logged_event_seq = event.seq;
          const std::string line = repo::FormatEventLine(event);
          if (event.severity == repo::DriftSeverity::Error) {
            LOG_ERROR << line;
          } else if (event.severity == repo::DriftSeverity::Warn) {
            LOG_WARNING << line;
          } else {
            LOG_INFO << line;
          }
        }
      } else {
        policy_engine->NoteSkippedPoll();
      }
      ++policy_poll_index;
    }

    // Wait for the polling interval (or a long time if polling is not
    // enabled). Will be woken if the server is exiting.
    std::unique_lock<std::mutex> lock(triton::server::signal_exit_mu_);
    std::chrono::seconds wait_timeout(
        (g_triton_params.repository_poll_secs_ == 0)
            ? 3600
            : g_triton_params.repository_poll_secs_);
    triton::server::signal_exit_cv_.wait_for(lock, wait_timeout);
  }

  // Stop the HTTP and gRPC endpoints, and update exit timeout.
  uint32_t exit_timeout_secs = g_triton_params.exit_timeout_secs_;
  StopEndpoints(&exit_timeout_secs);
  TRITONSERVER_ServerSetExitTimeout(server_ptr, exit_timeout_secs);

  TRITONSERVER_Error* stop_err = TRITONSERVER_ServerStop(server_ptr);

  // If unable to gracefully stop the server then Triton threads and
  // state are potentially in an invalid state, so just exit
  // immediately.
  if (stop_err != nullptr) {
    LOG_TRITONSERVER_ERROR(stop_err, "failed to stop server");
    exit(1);
  }

  // Stop gRPC and metrics endpoints that do not yet support exit timeout.
  StopEndpoints();

  // Stop tracing.
  StopTracing(&trace_manager);

#ifdef TRITON_ENABLE_ASAN
  // Can invoke ASAN before exit though this is typically not very
  // useful since there are many objects that are not yet destructed.
  //  __lsan_do_leak_check();
#endif  // TRITON_ENABLE_ASAN

  return 0;
}
