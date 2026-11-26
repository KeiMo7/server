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
#include <cctype>
#include <set>
#include <sstream>

namespace triton { namespace server { namespace repository {

using detail::Cursor;
using detail::ParseError;

std::string
Status::SanitizeMessage(const std::string& in)
{
  std::string out;
  out.reserve(std::min(in.size(), size_t(256)));
  for (char c : in) {
    if (c == '\n' || c == '\r') {
      out.push_back(' ');
    } else {
      out.push_back(c);
    }
    if (out.size() >= 256) {
      break;
    }
  }
  return out;
}

const char*
IndexStateName(VersionState state)
{
  switch (state) {
    case VersionState::Ready:
      return "READY";
    case VersionState::Loading:
      return "LOADING";
    case VersionState::Unavailable:
      return "UNAVAILABLE";
    case VersionState::Unloading:
      return "UNLOADING";
    case VersionState::Unknown:
    default:
      return "UNKNOWN";
  }
}

const char*
SeverityName(DriftSeverity severity)
{
  switch (severity) {
    case DriftSeverity::Error:
      return "error";
    case DriftSeverity::Warn:
      return "warn";
    case DriftSeverity::None:
    default:
      return "none";
  }
}

const char*
PolicyKindName(PolicyKind kind)
{
  switch (kind) {
    case PolicyKind::All:
      return "all";
    case PolicyKind::Specific:
      return "specific";
    case PolicyKind::Latest:
    default:
      return "latest";
  }
}

Status
ParseVersionPolicy(const std::string& config_text, VersionPolicy* policy)
{
  *policy = VersionPolicy{};  // default: Latest with n=1

  Cursor cur(config_text);
  cur.SkipSpace();

  // Locate the version_policy block; everything else in the file is
  // ignored on purpose.
  while (!cur.MatchWord("version_policy")) {
    if (cur.Eof()) {
      return Status::OK();  // no policy block: default Latest-1
    }
    cur.Match(static_cast<char>(config_text[cur.Pos()]));
  }

  cur.Match(':');  // optional "version_policy: {" spelling
  if (!cur.Match('{')) {
    return Status::Error(ParseError(&cur, "expected '{' after version_policy"));
  }
  auto optional_colon = [&cur]() { cur.Match(':'); };

  bool kind_seen = false;
  while (true) {
    cur.SkipSpace();
    if (cur.Match('}')) {
      break;
    }
    if (cur.Eof()) {
      return Status::Error(
          ParseError(&cur, "unbalanced braces in version_policy"));
    }
    if (kind_seen) {
      return Status::Error(
          ParseError(&cur, "version_policy accepts a single policy kind"));
    }

    if (cur.MatchWord("all")) {
      optional_colon();
      if (!cur.Match('{') || !cur.Match('}')) {
        return Status::Error(ParseError(&cur, "all policy takes no fields"));
      }
      policy->kind = PolicyKind::All;
      kind_seen = true;
    } else if (cur.MatchWord("latest")) {
      optional_colon();
      if (!cur.Match('{')) {
        return Status::Error(ParseError(&cur, "expected '{' after latest"));
      }
      uint32_t n = 0;
      bool field_seen = false;
      while (true) {
        cur.SkipSpace();
        if (cur.Match('}')) {
          break;
        }
        if (cur.Eof()) {
          return Status::Error(
              ParseError(&cur, "unbalanced braces in latest policy"));
        }
        if (field_seen || !cur.MatchWord("num_versions") ||
            !cur.Match(':')) {
          return Status::Error(
              ParseError(&cur, "latest policy expects num_versions: <n>"));
        }
        uint64_t parsed = 0;
        if (!cur.ParseUint64(&parsed) || parsed == 0 ||
            parsed > UINT32_MAX) {
          return Status::Error(ParseError(
              &cur, "num_versions must be a positive 32-bit integer"));
        }
        n = static_cast<uint32_t>(parsed);
        field_seen = true;
      }
      if (!field_seen) {
        return Status::Error(
            ParseError(&cur, "latest policy requires num_versions"));
      }
      policy->kind = PolicyKind::Latest;
      policy->num_versions = n;
      kind_seen = true;
    } else if (cur.MatchWord("specific")) {
      optional_colon();
      if (!cur.Match('{')) {
        return Status::Error(ParseError(&cur, "expected '{' after specific"));
      }
      bool field_seen = false;
      std::set<uint64_t> unique_versions;
      while (true) {
        cur.SkipSpace();
        if (cur.Match('}')) {
          break;
        }
        if (cur.Eof()) {
          return Status::Error(
              ParseError(&cur, "unbalanced braces in specific policy"));
        }
        if (field_seen || !cur.MatchWord("versions") || !cur.Match(':')) {
          return Status::Error(
              ParseError(&cur, "specific policy expects versions: [...]"));
        }
        if (!cur.Match('[')) {
          return Status::Error(ParseError(&cur, "expected '[' after versions"));
        }
        cur.SkipSpace();
        if (!cur.Match(']')) {
          while (true) {
            uint64_t v = 0;
            if (!cur.ParseUint64(&v) || v == 0) {
              return Status::Error(
                  ParseError(&cur, "versions must be positive integers"));
            }
            unique_versions.insert(v);
            cur.SkipSpace();
            if (cur.Match(',')) {
              continue;
            }
            if (!cur.Match(']')) {
              return Status::Error(
                  ParseError(&cur, "expected ',' or ']' in versions list"));
            }
            break;
          }
        }
        field_seen = true;
      }
      if (!field_seen) {
        return Status::Error(
            ParseError(&cur, "specific policy requires versions"));
      }
      policy->kind = PolicyKind::Specific;
      policy->versions.assign(unique_versions.begin(), unique_versions.end());
      kind_seen = true;
    } else {
      return Status::Error(
          ParseError(&cur, "unknown version_policy kind or field"));
    }
  }

  if (!kind_seen) {
    return Status::Error(ParseError(&cur, "empty version_policy block"));
  }
  return Status::OK();
}

std::vector<uint64_t>
ApplyVersionPolicy(
    const VersionPolicy& policy,
    const std::vector<VersionEntry>& candidates,
    std::vector<uint64_t>* missing)
{
  std::vector<uint64_t> available;
  available.reserve(candidates.size());
  for (const auto& entry : candidates) {
    available.push_back(entry.version);
  }
  std::sort(available.begin(), available.end());
  available.erase(
      std::unique(available.begin(), available.end()), available.end());

  std::vector<uint64_t> served;
  switch (policy.kind) {
    case PolicyKind::All:
      served = available;
      break;
    case PolicyKind::Latest: {
      uint32_t n = std::min<uint32_t>(
          policy.num_versions, static_cast<uint32_t>(available.size()));
      if (n > 0) {
        served.assign(available.end() - n, available.end());
      }
      break;
    }
    case PolicyKind::Specific: {
      std::set<uint64_t> avail_set(available.begin(), available.end());
      std::set<uint64_t> missing_set;
      for (uint64_t requested : policy.versions) {
        if (avail_set.count(requested) != 0) {
          served.push_back(requested);
        } else {
          missing_set.insert(requested);
        }
      }
      if (missing != nullptr) {
        missing->assign(missing_set.begin(), missing_set.end());
      }
      break;
    }
  }
  std::sort(served.begin(), served.end());
  return served;
}

RouteResult
RouteVersion(const ModelView& view, const std::string& requested)
{
  RouteResult result;

  if (requested.empty()) {
    if (view.served.empty()) {
      result.error =
          "Request for unknown model: '" + view.name +
          "' has no available versions";
      return result;
    }
    result.ok = true;
    result.version = view.served.back().version;  // numerically greatest
    return result;
  }

  for (char c : requested) {
    if (!std::isdigit(static_cast<unsigned char>(c))) {
      result.error = "invalid version";
      return result;
    }
  }

  uint64_t requested_version = 0;
  for (char c : requested) {
    uint8_t digit = static_cast<uint8_t>(c - '0');
    if (requested_version > (UINT64_MAX - digit) / 10ULL) {
      result.error = "invalid version";
      return result;
    }
    requested_version = requested_version * 10ULL + digit;
  }
  if (requested_version == 0) {
    result.error = "invalid version";
    return result;
  }

  for (const auto& vv : view.served) {
    if (vv.version == requested_version && vv.state == VersionState::Ready) {
      result.ok = true;
      result.version = requested_version;
      return result;
    }
  }

  if (view.served.empty()) {
    result.error =
        "Request for unknown model: '" + view.name +
        "' has no available versions";
  } else {
    result.error =
        "Request for unknown model: '" + view.name + "' version " +
        requested + " is not at ready state";
  }
  return result;
}

std::vector<RepositoryEvent>
FlattenEventChunks(const std::shared_ptr<const EventChunk>& newest)
{
  // The chain runs newest chunk -> older chunk, while within each chunk
  // events run oldest -> newest. Count first, fill from the back so the
  // result is globally oldest -> newest in linear time.
  size_t total = 0;
  for (auto cur = newest; cur; cur = cur->older) {
    total += cur->events.size();
  }
  std::vector<RepositoryEvent> out;
  out.resize(total);
  size_t cursor = total;
  for (auto cur = newest; cur; cur = cur->older) {
    const auto& evs = cur->events;
    cursor -= evs.size();
    for (size_t i = 0; i < evs.size(); ++i) {
      out[cursor + i] = evs[i];
    }
  }
  return out;
}

std::string
FormatEventLine(const RepositoryEvent& event)
{
  std::ostringstream oss;
  oss << "repo-policy: model="
      << (event.model.empty() ? std::string("-") : event.model)
      << " version=";
  if (event.version.has_value()) {
    oss << event.version.value();
  } else {
    oss << "-";
  }
  oss << " from=" << IndexStateName(event.from)
      << " to=" << IndexStateName(event.to)
      << " severity=" << SeverityName(event.severity)
      << " reason=" << event.reason;
  return oss.str();
}

// =========================================================================
// Engine

}}}  // namespace triton::server::repository
