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
// Filesystem side of the repository policy module: turns one or more
// repository directories into a RepositoryScan and computes the
// fingerprints used for two-round stability checks. Only the C++
// standard library is used, and model file contents are never read:
// fingerprints derive from names, sizes and modification times only.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "repository_policy.h"

namespace triton { namespace server { namespace repository {

// FNV-1a 64-bit over a byte buffer (exposed for deterministic testing).
uint64_t
Fnv1a64(const char* data, size_t size, uint64_t seed = 1469598103934665603ULL);

// FNV-1a digest synthesised from a version directory's (regular file
// count, total bytes, greatest last_write_time in nanoseconds). Model
// file contents are never read; any file change must surface through
// one of these three values.
uint64_t
VersionFingerprint(
    uint64_t file_count, uint64_t total_bytes, uint64_t mtime_ns);

// True when 'path' uses a cloud scheme (gs://, s3:// or as://) that
// this module cannot pre-check on the local filesystem.
bool
IsCloudRepositoryPath(const std::string& path);

// True when 'name' is a legal version directory name: pure decimal
// digits, no leading zero ("0" itself is illegal as a version and is
// rejected), and the parsed value fits in int64 and is non-zero.
// '*parsed' (when non-null) receives the parsed value on success.
bool
IsValidVersionDirectoryName(const std::string& name, uint64_t* parsed);

// Fingerprint of one version directory: FNV-1a over the tuple
// (regular file count, total bytes, greatest last_write_time in
// nanoseconds). Returns a non-OK Status if the directory cannot be
// iterated.
Status
ComputeVersionFingerprint(
    const std::string& version_dir, uint64_t* file_count,
    uint64_t* total_bytes, uint64_t* fingerprint);

// Locate the configuration file for one model following the Custom
// Model Configuration convention: when 'config_name' is non-empty,
// <model>/configs/<config_name>.pbtxt is tried first and
// <model>/config.pbtxt is the fallback. Returns an empty path and sets
// 'present=false' when neither exists.
Status
LocateModelConfig(
    const std::string& model_dir, const std::string& config_name,
    std::string* config_path, bool* present);

// Scan one model directory (<repo>/<model>). Version discovery only
// accepts <repo>/<model>/<version>/ directories whose names are pure,
// non-zero decimal numbers without leading zeros; everything else is
// recorded in 'skipped' (numeric names only) or ignored. Version
// directories containing no regular files are not candidates.
Status
ScanModel(
    const std::string& repo_root, const std::string& model_name,
    const std::string& config_name, ModelScan* model_scan);

// Scan one local repository root. Model directories are visited in
// sorted order so the cost model stays predictable; callers re-sort
// anyway when publishing snapshots. Any unreadable repository or model
// directory fails the whole scan (the engine then retains its previous
// snapshot).
Status
ScanRepository(
    const std::string& repo_root, const std::string& config_name,
    uint64_t scan_seq, RepositoryScan* scan);

// Scan several repository roots in the given order and merge the
// results into one scan. Model names appearing in more than one root
// are kept from the first root only, matching the server's
// first-wins uniqueness rule. Cloud paths are skipped with a warning
// rather than failing.
Status
ScanRepositories(
    const std::vector<std::string>& repo_roots,
    const std::string& config_name, uint64_t scan_seq,
    RepositoryScan* scan);

}}}  // namespace triton::server::repository
