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
// Internal helpers shared by the repository_policy translation units:
// the configuration-text cursor, the index-state vocabulary mapping
// and the two-round stability gate. Not part of the public module API.
#pragma once

#include <cctype>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>

#include "repository_policy.h"

namespace triton { namespace server { namespace repository { namespace detail {

// Minimal cursor over configuration text. '#' starts a line comment.
class Cursor {
 public:
  explicit Cursor(const std::string& text) : text_(text) {}

  void
  SkipSpace()
  {
    while (pos_ < text_.size()) {
      char c = text_[pos_];
      if (c == '#') {
        while (pos_ < text_.size() && text_[pos_] != '\n') {
          ++pos_;
        }
      } else if (std::isspace(static_cast<unsigned char>(c))) {
        ++pos_;
      } else {
        break;
      }
    }
  }

  bool
  Match(char c)
  {
    SkipSpace();
    if (pos_ < text_.size() && text_[pos_] == c) {
      ++pos_;
      return true;
    }
    return false;
  }

  bool
  MatchWord(const char* word)
  {
    SkipSpace();
    size_t len = std::strlen(word);
    if (text_.compare(pos_, len, word) == 0) {
      char next = (pos_ + len < text_.size()) ? text_[pos_ + len] : '\0';
      if (std::isalnum(static_cast<unsigned char>(next)) || next == '_') {
        return false;
      }
      pos_ += len;
      return true;
    }
    return false;
  }

  bool
  Eof()
  {
    SkipSpace();
    return pos_ >= text_.size();
  }

  size_t
  Pos() const
  {
    return pos_;
  }

  std::string
  Location() const
  {
    size_t line = 1;
    size_t col = 1;
    for (size_t i = 0; i < pos_ && i < text_.size(); ++i) {
      if (text_[i] == '\n') {
        ++line;
        col = 1;
      } else {
        ++col;
      }
    }
    std::ostringstream oss;
    oss << "line " << line << " col " << col;
    return oss.str();
  }

  // Parse an unsigned decimal integer with overflow detection.
  bool
  ParseUint64(uint64_t* out)
  {
    SkipSpace();
    size_t start = pos_;
    uint64_t value = 0;
    while (pos_ < text_.size() &&
           std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
      uint8_t digit = static_cast<uint8_t>(text_[pos_] - '0');
      if (value > (UINT64_MAX - digit) / 10ULL) {
        return false;
      }
      value = value * 10ULL + digit;
      ++pos_;
    }
    if (pos_ == start) {
      return false;
    }
    *out = value;
    return true;
  }

 private:
  const std::string& text_;
  size_t pos_{0};
};

inline std::string
ParseError(Cursor* cur, const std::string& detail)
{
  return "config_invalid at " + cur->Location() + ": " + detail;
}

inline VersionState
ParseIndexState(const std::string& text)
{
  if (text == "READY") {
    return VersionState::Ready;
  }
  if (text == "LOADING") {
    return VersionState::Loading;
  }
  if (text == "UNAVAILABLE") {
    return VersionState::Unavailable;
  }
  if (text == "UNLOADING") {
    return VersionState::Unloading;
  }
  return VersionState::Unknown;
}

// Two-round stability gate. The first observed value takes effect at
// once; afterwards a change is held as pending until two consecutive
// rounds agree, and a third distinct value is flagged unstable.
class StabilityGate {
 public:
  // When 'require_confirmation' is set, even the first observed value is
  // held as pending until the next round confirms it. This is used for
  // version directories that appear after the engine's first scan, so a
  // half-written new directory cannot become a served version.
  explicit StabilityGate(bool require_confirmation = false)
      : require_confirmation_(require_confirmation)
  {
  }

  uint64_t
  Update(uint64_t current, bool* unstable)
  {
    *unstable = false;
    if (!initialized_) {
      initialized_ = true;
      if (require_confirmation_) {
        pending_ = current;
        has_pending_ = true;
        return effective_;
      }
      effective_ = current;
      return effective_;
    }
    if (!has_pending_) {
      if (current != effective_) {
        pending_ = current;
        has_pending_ = true;
        replaced_ = false;
      }
      return effective_;
    }
    if (current == pending_) {
      if (has_pending_ && pending_ != effective_) {
        effective_ = pending_;
        *unstable = replaced_;
      }
      has_pending_ = false;
      return effective_;
    }
    pending_ = current;
    replaced_ = true;
    return effective_;
  }

  uint64_t
  Effective() const
  {
    return effective_;
  }

 private:
  uint64_t effective_{0};
  uint64_t pending_{0};
  bool initialized_{false};
  bool has_pending_{false};
  bool replaced_{false};
  bool require_confirmation_{false};
};

}}}}  // namespace triton::server::repository::detail
