#!/bin/bash

set -o nounset

git diff --exit-code > /dev/null
is_unstaged=$?

git diff --cached --exit-code > /dev/null
is_ncommittted=$?

: ${REQUIRE_STAGED=1}
: ${REQUIRE_COMMITTED=0}

if [ ! $is_ncommittted == 0 ] && [ $REQUIRE_STAGED == 1]; then
    echo "Cannot build an executable because there are unstaged changes."
    exit 1
fi
if [ ! $is_ncommittted == 0 ] && [ $REQUIRE_COMMITTED == 1 ]; then
    echo "Cannot build an executable because there are non-committed changes."
    exit 1
fi

