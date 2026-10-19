#!/bin/sh
# Run once after cloning: enables the attribution-stripping hook.
git config core.hooksPath .githooks
echo "core.hooksPath set to .githooks"
