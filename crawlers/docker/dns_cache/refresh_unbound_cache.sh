#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

unbound-control flush_zone .
unbound-control flush_bogus .
unbound-control flush_zone .
unbound-control flush_negative
unbound-control flush_infra all
cat ${SCRIPT_DIR}/tld.dump | unbound-control load_cache
