#!/usr/bin/env bash
set -euo pipefail

candidate_repo=${1:-.}
override=${2:-}

head_sha=$(git -C "$candidate_repo" rev-parse HEAD)

if [[ -n "$override" ]]; then
  git -C "$candidate_repo" rev-parse --verify "${override}^{commit}" >/dev/null
  printf '%s\n' "$override"
  exit 0
fi

# Use the highest version-like tag that is reachable from the candidate but
# does not point at the candidate itself. This makes a freshly tagged release
# compare against the preceding release automatically, while post-release
# development compares against the release it follows.
while IFS= read -r tag; do
  [[ -n "$tag" ]] || continue
  [[ "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$ ]] || continue
  tag_sha=$(git -C "$candidate_repo" rev-list -n 1 "$tag")
  if [[ "$tag_sha" != "$head_sha" ]]; then
    printf '%s\n' "$tag"
    exit 0
  fi
done < <(git -C "$candidate_repo" tag \
  --merged HEAD \
  --list 'v[0-9]*' \
  --sort=-version:refname)

printf 'No previous reachable release tag was found for candidate %s\n' "$head_sha" >&2
exit 1
