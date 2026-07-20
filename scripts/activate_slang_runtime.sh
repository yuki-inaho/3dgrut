#!/usr/bin/env bash

# Keep the Slang wrapper ahead of the incompatible upstream binary installed
# in the main Pixi environment. Other processes continue using the host glibc.
_3dgrut_repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="${_3dgrut_repo_dir}/scripts/slang-runtime-bin:${PATH}"
unset _3dgrut_repo_dir
