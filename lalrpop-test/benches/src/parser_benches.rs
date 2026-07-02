use criterion::{Criterion, criterion_group};
use std::fs;

use crate::src::parsers::json;
use crate::src::parsers::json_ref;
use crate::src::parsers::json_tail_call;

pub fn json_parse(c: &mut Criterion) {
    let parser = json::ValueParser::new();
    let json_file = fs::read_to_string("benches/data/512KB.json").unwrap();
    c.bench_function("Parse JSON", |b| {
        b.iter(|| {
            parser.parse(&json_file).unwrap();
        })
    });
}

pub fn json_ref_parse(c: &mut Criterion) {
    let parser = json_ref::ValueRefParser::new();
    let json_file = fs::read_to_string("benches/data/512KB.json").unwrap();
    c.bench_function("Parse JSON with Referenced Input", |b| {
        b.iter(|| {
            parser.parse(&json_file).unwrap();
        })
    });
}

pub fn json_tail_call_parse(c: &mut Criterion) {
    let parser = json_tail_call::ValueParser::new();
    let json_file = fs::read_to_string("benches/data/512KB.json").unwrap();
    c.bench_function("Parse JSON with Tail Call Parser", |b| {
        b.iter(|| {
            parser.parse(&json_file).unwrap();
        })
    });
}

criterion_group!(parser, json_parse, json_ref_parse, json_tail_call_parse);
