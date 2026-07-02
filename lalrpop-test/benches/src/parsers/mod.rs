use lalrpop_util::lalrpop_mod;

lalrpop_mod!(pub json, "parsers/json.rs");
pub mod json_val;
lalrpop_mod!(pub json_ref, "parsers/json_ref.rs");
lalrpop_mod!(pub json_tail_call, "parsers/json_tail_call.rs");
