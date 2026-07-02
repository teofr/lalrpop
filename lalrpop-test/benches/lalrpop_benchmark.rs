// Allows benchmarking guaranteed-tail-call parsers on nightly:
// RUSTFLAGS="--cfg lalrpop_tail_call_become" cargo +nightly bench
#![cfg_attr(lalrpop_tail_call_become, feature(explicit_tail_calls))]
#![cfg_attr(lalrpop_tail_call_become, allow(incomplete_features))]

use criterion::criterion_main;

mod src {
    pub mod compile_benches;
    pub mod parser_benches;
    pub mod parsers;
}

use crate::src::compile_benches::compile;
use crate::src::parser_benches::parser;

criterion_main!(compile, parser);
