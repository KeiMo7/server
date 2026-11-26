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

#include "repository_policy_scan.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <system_error>

namespace fs = std::filesystem;

namespace triton { namespace server { namespace repository {

namespace {

constexpr uint64_t kFnvOffsetBasis = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

uint64_t
FnvCombine(uint64_t hash, uint64_t value)
{
  for (size_t i = 0; i < sizeof(value); ++i) {
    hash ^= static_cast<uint8_t>((value >> (8 * i)) & 0xFFULL);
    hash *= kFnvPrime;
  }
  return hash;
}

bool
HasCloudScheme(const std::string& path, const std::string& scheme)
{
  return path.compare(0, scheme.size(), scheme) == 0;
}

}  // namespace

uint64_t
Fnv1a64(const char* data, size_t size, uint64_t seed)
{
  uint64_t hash = seed;
  for (size_t i = 0; i < size; ++i) {
    hash ^= static_cast<uint8_t>(data[i]);
    hash *= kFnvPrime;
  }
  return hash;
}

uint64_t
VersionFingerprint(uint64_t file_count, uint64_t total_bytes, uint64_t mtime_ns)
{
  uint64_t hash = kFnvOffsetBasis;
  hash = FnvCombine(hash, file_count);
  hash = FnvCombine(hash, total_bytes);
  hash = FnvCombine(hash, mtime_ns);
  return hash;
}

bool
IsCloudRepositoryPath(const std::string& path)
{
  return HasCloudScheme(path, "gs://") || HasCloudScheme(path, "s3://") ||
         HasCloudScheme(path, "as://");
}

bool
IsValidVersionDirectoryName(const std::string& name, uint64_t* parsed)
{
  if (name.empty()) {
    return false;
  }
  // Pure decimal digits only.
  for (char c : name) {
    if (c < '0' || c > '9') {
      return false;
    }
  }
  // Leading zero (including "0" itself) is rejected.
  if (name.size() > 1 && name[0] == '0') {
    return false;
  }
  uint64_t value = 0;
  for (char c : name) {
    uint8_t digit = static_cast<uint8_t>(c - '0');
    if (value > (UINT64_MAX - digit) / 10ULL) {
      return false;  // overflow
    }
    value = value * 10ULL + digit;
  }
  // Zero is not a legal version, and the value must fit in int64.
  if (value == 0 || value > static_cast<uint64_t>(
                                std::numeric_limits<int64_t>::max())) {
    return false;
  }
  if (parsed != nullptr) {
    *parsed = value;
  }
  return true;
}

Status
ComputeVersionFingerprint(
    const std::string& version_dir, uint64_t* file_count,
    uint64_t* total_bytes, uint64_t* fingerprint)
{
  uint64_t count = 0;
  uint64_t bytes = 0;
  uint64_t max_mtime_ns = 0;
  std::error_code ec;

  fs::recursive_directory_iterator it(
      version_dir, fs::directory_options::skip_permission_denied, ec);
  if (ec) {
    return Status::Error(
        "cannot iterate version directory '" + version_dir +
        "': " + ec.message());
  }
  fs::recursive_directory_iterator end;
  for (; it != end; it.increment(ec)) {
    if (ec) {
      return Status::Error(
          "failure while iterating version directory '" + version_dir +
          "': " + ec.message());
    }
    const fs::directory_entry& entry = *it;
    if (!entry.is_regular_file(ec) || ec) {
      ec.clear();
      continue;
    }
    ++count;
    uint64_t size = entry.file_size(ec);
    if (ec) {
      return Status::Error(
          "cannot stat file '" + entry.path().string() +
          "': " + ec.message());
    }
    bytes += size;
    auto mtime = entry.last_write_time(ec);
    if (ec) {
      return Status::Error(
          "cannot read mtime of '" + entry.path().string() +
          "': " + ec.message());
    }
    auto ns = mtime.time_since_epoch().count();
    uint64_t mtime_ns = ns < 0 ? 0 : static_cast<uint64_t>(ns);
    max_mtime_ns = std::max(max_mtime_ns, mtime_ns);
  }

  *file_count = count;
  *total_bytes = bytes;
  *fingerprint = VersionFingerprint(count, bytes, max_mtime_ns);
  return Status::OK();
}

Status
LocateModelConfig(
    const std::string& model_dir, const std::string& config_name,
    std::string* config_path, bool* present)
{
  *present = false;
  config_path->clear();
  std::error_code ec;

  if (!config_name.empty()) {
    fs::path custom = fs::path(model_dir) / "configs" /
                      (config_name + ".pbtxt");
    if (fs::exists(custom, ec) && !ec && fs::is_regular_file(custom, ec) &&
        !ec) {
      *config_path = custom.string();
      *present = true;
      return Status::OK();
    }
    ec.clear();
  }

  fs::path fallback = fs::path(model_dir) / "config.pbtxt";
  if (fs::exists(fallback, ec) && !ec &&
      fs::is_regular_file(fallback, ec) && !ec) {
    *config_path = fallback.string();
    *present = true;
  }
  return Status::OK();
}

namespace {

// Read a whole file into 'contents'. Files are read for configuration
// only; model files are never opened.
Status
ReadFile(const std::string& path, std::string* contents)
{
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    return Status::Error("cannot open configuration file '" + path + "'");
  }
  std::ostringstream oss;
  oss << in.rdbuf();
  *contents = oss.str();
  return Status::OK();
}

}  // namespace

Status
ScanModel(
    const std::string& repo_root, const std::string& model_name,
    const std::string& config_name, ModelScan* model_scan)
{
  model_scan->name = model_name;
  fs::path model_dir = fs::path(repo_root) / model_name;
  std::error_code ec;

  // ---- Configuration.
  std::string config_path;
  bool present = false;
  Status st =
      LocateModelConfig(model_dir.string(), config_name, &config_path, &present);
  if (!st.ok) {
    return st;
  }
  model_scan->config_present = present;
  if (present) {
    st = ReadFile(config_path, &model_scan->config_text);
    if (!st.ok) {
      return st;
    }
    model_scan->config_fingerprint = Fnv1a64(
        model_scan->config_text.data(), model_scan->config_text.size());
  }

  // ---- Version directories.
  fs::directory_iterator it(model_dir, ec);
  if (ec) {
    return Status::Error(
        "cannot iterate model directory '" + model_dir.string() +
        "': " + ec.message());
  }
  fs::directory_iterator end;
  std::vector<VersionEntry> entries;
  std::vector<uint64_t> skipped;
  for (; it != end; it.increment(ec)) {
    if (ec) {
      return Status::Error(
          "failure while iterating model directory '" + model_dir.string() +
          "': " + ec.message());
    }
    const fs::directory_entry& entry = *it;
    if (!entry.is_directory(ec) || ec) {
      ec.clear();
      continue;
    }
    const std::string version_name = entry.path().filename().string();
    uint64_t version = 0;
    if (!IsValidVersionDirectoryName(version_name, &version)) {
      // Keep a numeric record when the name was parseable at all.
      uint64_t raw = 0;
      bool all_digits = !version_name.empty();
      for (char c : version_name) {
        if (c < '0' || c > '9') {
          all_digits = false;
          break;
        }
      }
      if (all_digits) {
        raw = 0;
        bool overflow = false;
        for (char c : version_name) {
          uint8_t digit = static_cast<uint8_t>(c - '0');
          if (raw > (UINT64_MAX - digit) / 10ULL) {
            overflow = true;
            break;
          }
          raw = raw * 10ULL + digit;
        }
        if (!overflow) {
          skipped.push_back(raw);
        }
      }
      continue;
    }

    uint64_t file_count = 0;
    uint64_t total_bytes = 0;
    uint64_t fingerprint = 0;
    st = ComputeVersionFingerprint(
        entry.path().string(), &file_count, &total_bytes, &fingerprint);
    if (!st.ok) {
      return st;
    }
    if (file_count == 0) {
      // No regular files: not a candidate version, but it is a valid
      // numeric name so it is recorded as skipped.
      skipped.push_back(version);
      continue;
    }

    VersionEntry ve;
    ve.version = version;
    ve.relative_path = (fs::path(model_name) / version_name).string();
    ve.file_count = file_count;
    ve.total_bytes = total_bytes;
    ve.fingerprint = fingerprint;
    entries.push_back(std::move(ve));
  }

  std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
    return a.version < b.version;
  });
  std::sort(skipped.begin(), skipped.end());
  skipped.erase(std::unique(skipped.begin(), skipped.end()), skipped.end());
  model_scan->versions = std::move(entries);
  model_scan->skipped = std::move(skipped);
  return Status::OK();
}

Status
ScanRepository(
    const std::string& repo_root, const std::string& config_name,
    uint64_t scan_seq, RepositoryScan* scan)
{
  scan->scan_seq = scan_seq;
  std::error_code ec;
  if (!fs::exists(repo_root, ec) || ec) {
    return Status::Error(
        "model repository '" + repo_root + "' does not exist");
  }
  if (!fs::is_directory(repo_root, ec) || ec) {
    return Status::Error(
        "model repository '" + repo_root + "' is not a directory");
  }

  std::set<std::string> model_names;
  fs::directory_iterator it(repo_root, ec);
  if (ec) {
    return Status::Error(
        "cannot iterate model repository '" + repo_root +
        "': " + ec.message());
  }
  fs::directory_iterator end;
  for (; it != end; it.increment(ec)) {
    if (ec) {
      return Status::Error(
          "failure while iterating model repository '" + repo_root +
          "': " + ec.message());
    }
    if (!it->is_directory(ec) || ec) {
      ec.clear();
      continue;
    }
    model_names.insert(it->path().filename().string());
  }

  for (const std::string& name : model_names) {
    ModelScan model_scan;
    Status st = ScanModel(repo_root, name, config_name, &model_scan);
    if (!st.ok) {
      return st;
    }
    scan->models.push_back(std::move(model_scan));
  }
  return Status::OK();
}

Status
ScanRepositories(
    const std::vector<std::string>& repo_roots,
    const std::string& config_name, uint64_t scan_seq,
    RepositoryScan* scan)
{
  scan->scan_seq = scan_seq;
  scan->models.clear();
  std::set<std::string> seen_names;

  for (const std::string& root : repo_roots) {
    if (IsCloudRepositoryPath(root)) {
      // Cloud repositories are out of scope for local pre-checking; the
      // caller logs a warning and the server keeps handling them.
      continue;
    }
    RepositoryScan one;
    Status st = ScanRepository(root, config_name, scan_seq, &one);
    if (!st.ok) {
      return st;
    }
    for (ModelScan& ms : one.models) {
      if (!seen_names.insert(ms.name).second) {
        continue;  // first repository wins on duplicate model names
      }
      scan->models.push_back(std::move(ms));
    }
  }
  return Status::OK();
}

}}}  // namespace triton::server::repository
