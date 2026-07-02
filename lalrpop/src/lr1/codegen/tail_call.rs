//! A compiler from an LR(1) table to a tail-calling recursive ascent parser,
//! sometimes called a "recursive ascent pushdown" or defunctionalized-CPS
//! formulation of LR parsing.
//!
//! Like the classic recursive ascent backend (`ascent.rs`), each LR state is
//! compiled to its own function, so being "in" a state means the program
//! counter is inside that state's function and the ACTION dispatch is just the
//! `match` on the lookahead that we needed anyway. Unlike classic recursive
//! ascent, state functions never *return* to their caller to deliver a
//! nonterminal. Instead:
//!
//! * The parse stack is an explicit `Vec` of `(spanned symbol, continuation)`
//!   entries carried in a `Parser` struct. The continuation stored alongside a
//!   symbol is the GOTO row -- as a function pointer -- of the state that
//!   pushed the symbol (i.e. the state that is exposed if a reduction pops
//!   down to that symbol).
//!
//! * Every transition (shift, reduce dispatch, goto) is a call in tail
//!   position with the exact same signature, so the native call stack does
//!   not need to grow at all: a reduction pops its handle from the explicit
//!   stack, runs the action code, and transfers control to the continuation
//!   found on the deepest popped entry.
//!
//! * All parser functions take `(user parameters.., &mut Parser)` and return
//!   `()`; the lookahead and the final result are fields of the `Parser`
//!   struct. This keeps every argument and the return value in registers --
//!   a by-value argument or return type too large for registers is passed
//!   through the caller's stack frame, which makes the call ineligible for
//!   tail-call optimization.
//!
//! * The parser functions are `#[inline(never)]`, and both the pop+downcast
//!   sequences and the user's action code are fenced off behind small helper
//!   functions (`pop_VariantN` / `call_actionN`). Each of these guards a
//!   different way of losing the sibling-call optimization: parser functions
//!   inlined into one another leave their tail calls stranded in the middle
//!   of the merged function; inline downcasts make rustc emit drop-flagged
//!   unwind cleanups that give the final call an unwind edge; and inlined
//!   action code can contain operations that LLVM treats as escaping the
//!   caller's stack (for example a panic path passing a local by reference),
//!   which suppresses tail-call marking for the whole function.
//!
//! * No non-scalar named local is in scope at any transition: the lookahead
//!   is dispatched on by reference, values move through single-statement
//!   temporaries, and reduce bodies run inside inner blocks with the
//!   transition after the block. A local still in scope at the transition
//!   gets its `StorageDead` -- and hence, if it survives to a stack slot,
//!   its `llvm.lifetime.end` -- emitted between the call and the return,
//!   which stops the backend from turning the call into a jump. Relying on
//!   SROA to dissolve such locals works only by luck; the block structure
//!   makes it deterministic.
//!
//! Two of the optimizations from the tc_args design are derived mechanically
//! from the automaton (sections 3.4-3.6 of that report):
//!
//! * A shift into a *reduce-only* state (single production, no shifts, no
//!   gotos) is fused with that reduction: the shifted terminal stays in a
//!   local -- it never touches the explicit stack and is never wrapped in
//!   the `Symbol` enum -- while the fused code performs the target state's
//!   lookahead check and error reporting verbatim.
//!
//! * Reductions statically resolve their surviving state where possible:
//!   `StateGraph::trace_back` computes which states can be exposed by
//!   popping the handle, and when there is exactly one, the pushed
//!   continuation and the goto transition become compile-time constants
//!   instead of an indirect call through the stored continuation. Fusion
//!   compounds this: the popped prefix is anchored at the *shifting* state,
//!   which resolves uniquely more often (for a fused unit production it
//!   always does).
//!
//! All transitions are wrapped in a generated `transition!` macro. By default
//! it expands to a plain `return f(...)`, which LLVM reliably compiles to a
//! jump (a sibling tail call) in optimized builds; in debug builds the native
//! stack then grows with the number of transitions, i.e. with input length.
//! When the generated crate is compiled with `--cfg lalrpop_tail_call_become`
//! (a nightly compiler with `#![feature(explicit_tail_calls)]` enabled at the
//! crate root), the macro instead expands to `become f(...)`, which makes the
//! O(1)-native-stack property a compile-time guarantee even in debug builds.
//! The unified function signature shared by the state, goto, and reduce
//! functions exists precisely because `become` requires caller and callee
//! signatures to match exactly.
//!
//! Error recovery (the `!` symbol) is not supported, just as with the classic
//! recursive ascent backend.

use crate::collections::{Entry, Map, Multimap};
use crate::grammar::repr::{
    Grammar, NonterminalString, Production, Symbol, TerminalString, TypeParameter, TypeRepr,
    Visibility, WhereClause,
};
use crate::lr1::core::*;
use crate::lr1::lookahead::Token;
use crate::lr1::state_graph::StateGraph;
use crate::rust::RustWrite;
use crate::tls::Tls;
use crate::util::Sep;
use std::io::{self, Write};

use super::base::CodeGenerator;

/// The `--cfg` flag under which the generated code uses the (nightly-only)
/// `become` keyword for its transitions instead of plain `return`.
const BECOME_CFG: &str = "lalrpop_tail_call_become";

pub fn compile<'grammar, W: Write>(
    grammar: &'grammar Grammar,
    user_start_symbol: NonterminalString,
    start_symbol: NonterminalString,
    states: &[Lr1State<'grammar>],
    action_module: &str,
    out: &mut RustWrite<W>,
) -> io::Result<()> {
    let graph = StateGraph::new(states);
    let mut tail_call = CodeGenerator::new_tail_call(
        grammar,
        user_start_symbol,
        start_symbol,
        &graph,
        states,
        action_module,
        out,
    );
    tail_call.write()
}

struct TailCall<'ascent, 'grammar> {
    /// the shift/goto edges of the automaton; used to statically resolve
    /// which state survives a reduction (see `reduce_resolution`)
    graph: &'ascent StateGraph,

    /// type parameters for the `Symbol` type
    symbol_type_params: Vec<TypeParameter>,

    symbol_where_clauses: Vec<WhereClause>,

    /// a list of each nonterminal in some specific order; the position of a
    /// nonterminal in this list is the discriminant communicated to the goto
    /// functions through the `reduced_nt` field of the parser struct
    all_nonterminals: Vec<NonterminalString>,

    /// unique index assigned to each production, used to name the shared
    /// `reduceN` functions
    reduce_indices: Map<&'grammar Production, usize>,

    variant_names: Map<Symbol, String>,
    variants: Map<TypeRepr, String>,
}

impl<'ascent, 'grammar, W: Write> CodeGenerator<'ascent, 'grammar, W, TailCall<'ascent, 'grammar>> {
    #[allow(clippy::too_many_arguments)]
    fn new_tail_call(
        grammar: &'grammar Grammar,
        user_start_symbol: NonterminalString,
        start_symbol: NonterminalString,
        graph: &'ascent StateGraph,
        states: &'ascent [Lr1State<'grammar>],
        action_module: &str,
        out: &'ascent mut RustWrite<W>,
    ) -> Self {
        let (symbol_type_params, symbol_where_clauses) =
            Self::filter_type_parameters_and_where_clauses(
                grammar,
                grammar
                    .types
                    .nonterminal_types()
                    .into_iter()
                    .chain(grammar.types.terminal_types()),
            );

        let reduce_indices: Map<&'grammar Production, usize> = grammar
            .nonterminals
            .values()
            .flat_map(|nt| &nt.productions)
            .zip(0..)
            .collect();

        CodeGenerator::new(
            grammar,
            user_start_symbol,
            start_symbol,
            states,
            out,
            false,
            action_module,
            TailCall {
                graph,
                symbol_type_params,
                symbol_where_clauses,
                all_nonterminals: grammar.nonterminals.keys().cloned().collect(),
                reduce_indices,
                variant_names: Map::new(),
                variants: Map::new(),
            },
        )
    }

    fn write(&mut self) -> io::Result<()> {
        self.write_parse_mod(|this| {
            this.write_transition_macro()?;
            this.write_value_type_defn()?;
            this.write_goto_type_defn()?;
            this.write_parser_type_defn()?;
            this.write_helper_fns()?;
            this.write_pop_fns()?;
            this.write_action_shims()?;
            this.write_parser_fn()?;
            for i in 0..this.states.len() {
                this.write_state_fn(StateIndex(i))?;
            }
            for i in 0..this.states.len() {
                this.write_goto_fn(StateIndex(i))?;
            }
            this.write_reduce_fns()?;
            Ok(())
        })
    }

    /// The `transition!` macro wrapping every tail call. The `become`
    /// definition lives inside a macro so that stable compilers never have to
    /// *parse* a `become` expression (the feature is gated at parse time,
    /// even under `#[cfg]`); an unexpanded macro body is only lexed.
    fn write_transition_macro(&mut self) -> io::Result<()> {
        rust!(
            self.out,
            "// Every state/goto/reduce transition below is in"
        );
        rust!(self.out, "// tail position and shares one signature. With");
        rust!(
            self.out,
            "// `--cfg {}` (nightly, requires the crate",
            BECOME_CFG
        );
        rust!(
            self.out,
            "// to enable `#![feature(explicit_tail_calls)]`), transitions"
        );
        rust!(
            self.out,
            "// use `become` and the parser is guaranteed to use O(1) native"
        );
        rust!(
            self.out,
            "// stack. Otherwise they are sibling calls that LLVM turns into"
        );
        rust!(self.out, "// jumps in optimized builds.");
        rust!(self.out, "#[allow(unused_macros)]");
        rust!(self.out, "#[cfg({})]", BECOME_CFG);
        rust!(
            self.out,
            "macro_rules! {}transition {{ ($call:expr) => {{ become $call }}; }}",
            self.prefix
        );
        rust!(self.out, "#[allow(unused_macros)]");
        rust!(self.out, "#[cfg(not({}))]", BECOME_CFG);
        rust!(
            self.out,
            "macro_rules! {}transition {{ ($call:expr) => {{ return $call }}; }}",
            self.prefix
        );
        rust!(self.out, "");
        Ok(())
    }

    fn write_value_type_defn(&mut self) -> io::Result<()> {
        // sometimes some of the variants are not used, particularly
        // if we are generating multiple parsers from the same file:
        rust!(self.out, "#[allow(dead_code)]");
        rust!(
            self.out,
            "enum {}Symbol<{}>",
            self.prefix,
            Sep(", ", &self.custom.symbol_type_params),
        );

        if !self.custom.symbol_where_clauses.is_empty() {
            rust!(
                self.out,
                " where {}",
                Sep(", ", &self.custom.symbol_where_clauses),
            );
        }

        rust!(self.out, " {{");

        // make one variant per terminal
        for term in &self.grammar.terminals.all {
            let ty = self.types.terminal_type(term).clone();
            self.add_symbol_variant(Symbol::Terminal(term.clone()), ty)?;
        }

        // make one variant per nonterminal
        for nt in self.grammar.nonterminals.keys() {
            let ty = self.types.nonterminal_type(nt).clone();
            self.add_symbol_variant(Symbol::Nonterminal(nt.clone()), ty)?;
        }

        rust!(self.out, "}}");
        Ok(())
    }

    fn add_symbol_variant(&mut self, symbol: Symbol, ty: TypeRepr) -> io::Result<()> {
        let len = self.custom.variants.len();
        let name = match self.custom.variants.entry(ty.clone()) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let name = format!("Variant{len}");
                rust!(self.out, "{}({}),", name, ty);
                entry.insert(name)
            }
        };
        self.custom.variant_names.insert(symbol, name.clone());
        Ok(())
    }

    /// The continuation type: a newtype around the fn pointer for a state's
    /// GOTO row. (The newtype breaks the type recursion: an entry contains a
    /// `Goto` which contains a fn taking `&mut Parser` which contains a `Vec`
    /// of entries.)
    fn write_goto_type_defn(&mut self) -> io::Result<()> {
        let type_decls = self.parser_type_decls();
        let type_args = self.parser_type_args();
        let where_clauses = self.parser_where_clauses();
        rust!(
            self.out,
            "struct {}Goto<{}>(fn({})) where {};",
            self.prefix,
            type_decls,
            self.unified_fn_params(),
            where_clauses,
        );
        // Fn pointers are `Copy`, but `derive` would add unwanted bounds on
        // the type parameters, so implement by hand:
        rust!(
            self.out,
            "impl<{}> Copy for {}Goto<{}> where {} {{}}",
            type_decls,
            self.prefix,
            type_args,
            where_clauses,
        );
        rust!(
            self.out,
            "impl<{}> Clone for {}Goto<{}> where {} {{ fn clone(&self) -> Self {{ *self }} }}",
            type_decls,
            self.prefix,
            type_args,
            where_clauses,
        );
        Ok(())
    }

    fn write_parser_type_defn(&mut self) -> io::Result<()> {
        rust!(
            self.out,
            "struct {}Parser<{}> where {} {{",
            self.prefix,
            self.parser_type_decls(),
            self.parser_where_clauses(),
        );
        rust!(self.out, "{}tokens: {}TOKENS,", self.prefix, self.prefix);
        // The lookahead lives in the parser struct rather than being passed
        // as an argument: the triple can be large, and a large by-value
        // argument is passed through the caller's stack frame, which would
        // defeat the sibling-call optimization that keeps the native stack
        // flat in optimized builds.
        rust!(
            self.out,
            "{}lookahead: Option<{}>,",
            self.prefix,
            self.triple_type()
        );
        // Each entry pairs the semantic value of a stacked symbol with the
        // GOTO continuation of the state that pushed it, i.e. the state that
        // becomes current again if a reduction pops down to this entry.
        rust!(
            self.out,
            "{}stack: alloc::vec::Vec<({}, {}Goto<{}>)>,",
            self.prefix,
            self.spanned_symbol_type(),
            self.prefix,
            self.parser_type_args(),
        );
        // Which nonterminal was just reduced; the goto functions dispatch on
        // this. (It is a field rather than an argument so that all parser
        // functions can share one signature, as `become` requires.)
        rust!(self.out, "{}reduced_nt: usize,", self.prefix);
        // The final outcome of the parse. Like the lookahead, this lives in
        // the parser struct instead of flowing through the return values:
        // the success and error types can be large, and a large return value
        // is written through a hidden pointer into the caller's stack frame,
        // which would defeat the sibling-call optimization. Every parser
        // function returns `()`; whoever ends the parse (accept or error)
        // fills this in.
        rust!(
            self.out,
            "{}result: Option<{}>,",
            self.prefix,
            self.result_type()
        );
        rust!(
            self.out,
            "{}phantom: {},",
            self.prefix,
            self.phantom_data_type()
        );
        rust!(self.out, "}}");
        Ok(())
    }

    fn write_helper_fns(&mut self) -> io::Result<()> {
        // next_token
        let tokens_bound = self.tokens_bound();
        let parameters = vec![
            format!("{}tokens: &mut {}TOKENS", self.prefix, self.prefix),
            format!("_: {}", self.phantom_data_type()),
        ];
        let return_type = format!(
            "Result<Option<{}>, {}>",
            self.triple_type(),
            self.types.parse_error_type()
        );
        // `#[inline]` matters beyond performance: it makes every codegen
        // unit instantiate this helper locally, so it is reliably inlined
        // into the parser functions. A cross-codegen-unit call returning
        // this aggregate through a hidden pointer right before a transition
        // can otherwise stop LLVM from marking the transition as a tail
        // call.
        rust!(self.out, "#[inline]");
        self.out
            .fn_header(&Visibility::Priv, format!("{}next_token", self.prefix))
            .with_type_parameters(&self.grammar.type_parameters)
            .with_type_parameters(Some(tokens_bound))
            .with_where_clauses(&self.grammar.where_clauses)
            .with_parameters(parameters)
            .with_return_type(return_type)
            .emit()?;
        rust!(self.out, "{{");
        rust!(self.out, "match {}tokens.next() {{", self.prefix);
        rust!(
            self.out,
            "Some(Ok({}v)) => Ok(Some({}v)),",
            self.prefix,
            self.prefix
        );
        rust!(
            self.out,
            "Some(Err({}e)) => Err({}e),",
            self.prefix,
            self.prefix
        );
        rust!(self.out, "None => Ok(None),");
        rust!(self.out, "}}");
        rust!(self.out, "}}");
        rust!(self.out, "");

        // symbol_type_mismatch
        rust!(self.out, "#[inline(never)]");
        rust!(self.out, "#[allow(dead_code)]");
        rust!(self.out, "fn {}symbol_type_mismatch() -> ! {{", self.prefix);
        rust!(self.out, "panic!(\"symbol type mismatch\")");
        rust!(self.out, "}}");
        Ok(())
    }

    /// One `pop_VariantN` helper per symbol type: pops the top stack entry
    /// and downcasts it to a spanned value of that type (panicking on a
    /// type mismatch, which would be a bug in the generated parser). See
    /// `emit_pop_handle` for why popping and downcasting are fused into a
    /// helper function instead of being emitted inline in the reduce code.
    fn write_pop_fns(&mut self) -> io::Result<()> {
        for (ty, variant_name) in self.custom.variants.clone() {
            let tokens_bound = self.tokens_bound();
            let parameters = vec![format!(
                "{}stack: &mut alloc::vec::Vec<({}, {}Goto<{}>)>",
                self.prefix,
                self.spanned_symbol_type(),
                self.prefix,
                self.parser_type_args()
            )];
            let loc_type = self.types.terminal_loc_type();
            let return_type = format!(
                "(({}, {}, {}), {}Goto<{}>)",
                loc_type,
                ty,
                loc_type,
                self.prefix,
                self.parser_type_args()
            );
            rust!(self.out, "#[allow(dead_code)]");
            // see `next_token` for why the `#[inline]` is load-bearing
            rust!(self.out, "#[inline]");
            self.out
                .fn_header(
                    &Visibility::Priv,
                    format!("{}pop_{}", self.prefix, variant_name),
                )
                .with_type_parameters(&self.grammar.type_parameters)
                .with_type_parameters(Some(tokens_bound))
                .with_where_clauses(&self.grammar.where_clauses)
                .with_parameters(parameters)
                .with_return_type(return_type)
                .emit()?;
            rust!(self.out, "{{");
            rust!(self.out, "match {}stack.pop() {{", self.prefix);
            rust!(
                self.out,
                "Some((({}l, {}Symbol::{}({}v), {}r), {}g)) => (({}l, {}v, {}r), {}g),",
                self.prefix,
                self.prefix,
                variant_name,
                self.prefix,
                self.prefix,
                self.prefix,
                self.prefix,
                self.prefix,
                self.prefix,
                self.prefix
            );
            rust!(self.out, "_ => {}symbol_type_mismatch(),", self.prefix);
            rust!(self.out, "}}");
            rust!(self.out, "}}");
            rust!(self.out, "");
        }
        Ok(())
    }

    // Generates the `parse_Foo` entry point: build the parser struct, prime
    // the lookahead, and enter state 0.
    fn write_parser_fn(&mut self) -> io::Result<()> {
        self.start_parser_fn()?;
        self.define_tokens()?;

        rust!(
            self.out,
            "let mut {}parser = {}Parser {{",
            self.prefix,
            self.prefix
        );
        rust!(self.out, "{}tokens,", self.prefix);
        rust!(self.out, "{}lookahead: None,", self.prefix);
        rust!(
            self.out,
            "{}stack: alloc::vec::Vec::with_capacity(16),",
            self.prefix
        );
        rust!(self.out, "{}reduced_nt: 0,", self.prefix);
        rust!(self.out, "{}result: None,", self.prefix);
        rust!(
            self.out,
            "{}phantom: {},",
            self.prefix,
            self.phantom_data_expr()
        );
        rust!(self.out, "}};");
        rust!(
            self.out,
            "{}parser.{}lookahead = {}next_token(&mut {}parser.{}tokens, {})?;",
            self.prefix,
            self.prefix,
            self.prefix,
            self.prefix,
            self.prefix,
            self.phantom_data_expr()
        );
        rust!(
            self.out,
            "{}state0({}&mut {}parser);",
            self.prefix,
            self.grammar.user_parameter_refs(),
            self.prefix
        );
        // every terminal path (accept or error) fills in the result before
        // returning
        rust!(
            self.out,
            "{}parser.{}result.expect(\"tail call parser did not produce a result\")",
            self.prefix,
            self.prefix
        );

        self.end_parser_fn()
    }

    /// Writes the function corresponding to a given state. On entry, all the
    /// symbols making up the state's known stack prefix are on the explicit
    /// stack; the function dispatches on the lookahead, either shifting
    /// (push + tail call to the target state) or reducing (tail call to the
    /// production's shared reduce function).
    fn write_state_fn(&mut self, this_index: StateIndex) -> io::Result<()> {
        let this_state = &self.states[this_index.0];

        rust!(self.out, "");

        // Leave a comment explaining what this state is.
        if Tls::session().emit_comments {
            rust!(self.out, "// State {}", this_index.0);
            rust!(self.out, "//");
            for item in this_state.items.vec.iter() {
                rust!(self.out, "//     {:?}", item);
            }
            rust!(self.out, "//");
            for (terminal, action) in &this_state.shifts {
                rust!(self.out, "//   {:?} -> {:?}", terminal, action);
            }
            for &(ref tokens, action) in &this_state.reductions {
                rust!(self.out, "//   {:?} -> {:?}", tokens, action);
            }
            rust!(self.out, "//");
            for (nt, state) in &this_state.gotos {
                rust!(self.out, "//     {:?} -> {:?}", nt, state);
            }
        }

        // a state all of whose shift-predecessors fuse its reduction into
        // their own bodies is never entered
        rust!(self.out, "#[allow(dead_code)]");
        self.emit_parser_fn_header(format!("{}state{}", self.prefix, this_index.0))?;

        // Dispatch on the lookahead BY REFERENCE, binding nothing. Keeping
        // an owned copy of the lookahead as a local would leave a stack slot
        // in scope at every transition below; MIR then emits the slot's
        // StorageDead between the transition call and the return, and if
        // the slot survives to a stack allocation (SROA is best-effort),
        // the resulting `llvm.lifetime.end` after the call stops the
        // backend from compiling the transition as a tail call. The same
        // discipline -- no non-scalar named local in scope at a transition
        // -- shapes all the arms below.
        rust!(
            self.out,
            "match &{}parser.{}lookahead {{",
            self.prefix,
            self.prefix
        );

        // first emit shifts:
        for (terminal, &next_index) in &this_state.shifts {
            if let Some(production) = self.fused_production(next_index) {
                self.emit_fused_shift_reduce(this_index, terminal, next_index, production)?;
                rust!(self.out, "}}");
                continue;
            }

            let dispatch_pattern = self.match_terminal_pattern(terminal);
            rust!(self.out, "Some({}) => {{", dispatch_pattern);

            // push the shifted terminal, paired with our own goto row, and
            // transfer control to the target state. The token is taken,
            // rewrapped as a Symbol, and pushed in a single statement, so
            // the value only lives in statement temporaries.
            let (pattern, content) = self.terminal_pattern_and_content(terminal);
            rust!(
                self.out,
                "match {}parser.{}lookahead.take() {{",
                self.prefix,
                self.prefix
            );
            rust!(
                self.out,
                "Some(({}loc1, {}, {}loc2)) => {}parser.{}stack.push((({}loc1, {}Symbol::{}({}), {}loc2), {}Goto({}goto{}))),",
                self.prefix,
                pattern,
                self.prefix,
                self.prefix,
                self.prefix,
                self.prefix,
                self.prefix,
                self.variant_name_for_symbol(&Symbol::Terminal(terminal.clone())),
                content,
                self.prefix,
                self.prefix,
                self.prefix,
                this_index.0
            );
            rust!(self.out, "_ => unreachable!(),");
            rust!(self.out, "}}");
            self.emit_advance_lookahead()?;
            rust!(
                self.out,
                "{}transition!({}state{}({}{}parser))",
                self.prefix,
                self.prefix,
                next_index.0,
                self.grammar.user_parameter_refs(),
                self.prefix
            );
            rust!(self.out, "}}");
        }

        // now emit reduces. It frequently happens that many tokens
        // trigger the same reduction, so group these by the
        // production that we are going to be reducing.
        let reductions: Multimap<_, Vec<_>> = this_state
            .reductions
            .iter()
            .flat_map(|&(ref tokens, production)| tokens.iter().map(move |t| (production, t)))
            .collect();
        for (production, tokens) in reductions {
            for (index, token) in tokens.iter().enumerate() {
                let pattern = match *token {
                    Token::Terminal(ref s) => format!("Some({})", self.match_terminal_pattern(s)),
                    Token::Error => {
                        panic!("Error recovery is not implemented for tail call parsers")
                    }
                    Token::Eof => "None".to_string(),
                };
                if index < tokens.len() - 1 {
                    rust!(self.out, "{} |", pattern);
                } else {
                    rust!(self.out, "{} => {{", pattern);
                }
            }

            // the lookahead is not consumed by a reduction: it stays in the
            // parser struct untouched
            if production.nonterminal == self.start_symbol {
                self.emit_accept(production)?;
            } else if production.symbols.is_empty() {
                self.emit_empty_reduce(this_index, production)?;
            } else {
                // the reduce code depends only on the production and on
                // the statically-resolved survivor (if any), so it is
                // shared between all the reduce sites that agree on both
                let resolution = self.reduce_resolution(this_index, &production.symbols);
                rust!(
                    self.out,
                    "{}transition!({}({}{}parser))",
                    self.prefix,
                    self.reduce_fn_name(production, resolution),
                    self.grammar.user_parameter_refs(),
                    self.prefix
                );
            }

            rust!(self.out, "}}");
        }

        // if we hit this, the next token is not recognized, so generate an error
        rust!(self.out, "_ => {{");
        self.emit_error(this_state, None)?;
        rust!(self.out, "}}"); // Wildcard match case

        rust!(self.out, "}}"); // match
        rust!(self.out, "}}"); // fn

        Ok(())
    }

    /// The terminals which would have resulted in a successful parse in this
    /// state, and the error to return for anything else. `eof_location` is
    /// the expression giving the error location on EOF: by default the end
    /// of the last symbol on the stack, but fused shift-reduces must use the
    /// end of the shifted terminal they kept as a local instead.
    fn emit_error(
        &mut self,
        this_state: &Lr1State<'_>,
        eof_location: Option<String>,
    ) -> io::Result<()> {
        let successful_terminals = self.grammar.terminals.all.iter().filter(|&terminal| {
            this_state.shifts.contains_key(terminal)
                || this_state
                    .reductions
                    .iter()
                    .any(|(t, _)| t.contains(&Token::Terminal(terminal.clone())))
        });

        rust!(self.out, "#[allow(clippy::needless_raw_string_hashes)]");
        rust!(self.out, "let {}expected = alloc::vec![", self.prefix);
        for terminal in successful_terminals {
            // Try to avoid terminals escaping
            rust!(self.out, "r###\"{}\"###.to_string(),", terminal);
        }
        rust!(self.out, "];");

        rust!(
            self.out,
            "{}parser.{}result = Some(Err(match {}parser.{}lookahead.take() {{",
            self.prefix,
            self.prefix,
            self.prefix,
            self.prefix
        );
        rust!(self.out, "Some({}token) => {{", self.prefix);
        rust!(
            self.out,
            "{}lalrpop_util::ParseError::UnrecognizedToken {{",
            self.prefix
        );
        rust!(self.out, "token: {}token,", self.prefix);
        rust!(self.out, "expected: {}expected,", self.prefix);
        rust!(self.out, "}}");
        rust!(self.out, "}}");
        rust!(self.out, "None => {{");
        match eof_location {
            Some(location) => {
                rust!(self.out, "let {}location = {};", self.prefix, location);
            }
            None => {
                // find the location of the last symbol on the stack, if any
                rust!(
                    self.out,
                    "let {}location = match {}parser.{}stack.last() {{",
                    self.prefix,
                    self.prefix,
                    self.prefix
                );
                rust!(
                    self.out,
                    "Some({}entry) => ({}entry.0).2.clone(),",
                    self.prefix,
                    self.prefix
                );
                rust!(self.out, "None => Default::default(),");
                rust!(self.out, "}};");
            }
        }
        rust!(
            self.out,
            "{}lalrpop_util::ParseError::UnrecognizedEof {{",
            self.prefix
        );
        rust!(self.out, "location: {}location,", self.prefix);
        rust!(self.out, "expected: {}expected,", self.prefix);
        rust!(self.out, "}}");
        rust!(self.out, "}}");
        rust!(self.out, "}}));");
        rust!(self.out, "return;");
        Ok(())
    }

    /// Fetches the next token into `parser.lookahead`, routing lexer errors
    /// into the parse result.
    fn emit_advance_lookahead(&mut self) -> io::Result<()> {
        rust!(
            self.out,
            "match {}next_token(&mut {}parser.{}tokens, {}) {{",
            self.prefix,
            self.prefix,
            self.prefix,
            self.phantom_data_expr()
        );
        rust!(
            self.out,
            "Ok({}la) => {}parser.{}lookahead = {}la,",
            self.prefix,
            self.prefix,
            self.prefix,
            self.prefix
        );
        rust!(
            self.out,
            "Err({}e) => {{ {}parser.{}result = Some(Err({}e)); return; }}",
            self.prefix,
            self.prefix,
            self.prefix,
            self.prefix
        );
        rust!(self.out, "}}");
        Ok(())
    }

    /// Fuses a shift into a reduce-only state (report sections 3.4-3.6): the
    /// target state would only consult one lookahead token and pop a handle
    /// whose top we are about to push, so instead the shifted terminal is
    /// kept as a local -- it never touches the explicit stack, and is never
    /// wrapped in the `Symbol` enum -- and the target state's reduction is
    /// performed right here. A further payoff is that the popped handle
    /// prefix is anchored at *this* state rather than being traced from the
    /// target state, so the surviving state is more often statically known
    /// and the transition devirtualizes (see `reduce_resolution`).
    ///
    /// The lookahead consultation and the error report are exactly the ones
    /// the target state's function would have produced, except that on EOF
    /// the error location is taken from the local terminal (which is where
    /// the target state's stack top would have ended).
    fn emit_fused_shift_reduce(
        &mut self,
        this_index: StateIndex,
        terminal: &TerminalString,
        target_index: StateIndex,
        production: &'grammar Production,
    ) -> io::Result<()> {
        let len = production.symbols.len();
        assert_eq!(
            production.symbols[len - 1],
            Symbol::Terminal(terminal.clone())
        );

        if Tls::session().emit_comments {
            rust!(
                self.out,
                "// shifting {:?} into state {} is fused with reducing `{:?}`",
                terminal,
                target_index.0,
                production
            );
        }

        // arm head: dispatch on the terminal by reference
        let dispatch_pattern = self.match_terminal_pattern(terminal);
        rust!(self.out, "Some({}) => {{", dispatch_pattern);

        // the handle prefix below the local terminal is popped from the
        // stack; it is anchored at the current state (which is why the
        // survivor resolves statically more often than at an unfused
        // reduce site)
        let resolution = self.reduce_resolution(this_index, &production.symbols[..len - 1]);

        // The whole fused reduce runs inside an inner block, so that every
        // stack slot it creates (the terminal itself, the popped symbols,
        // the action result) is dead -- StorageDead emitted, lifetime over
        // -- before the transition that follows the block. See the
        // dispatch comment in `write_state_fn`: a slot still in scope at
        // the transition would end up with its `llvm.lifetime.end` after
        // the call, preventing the tail call. For a dynamic survivor the
        // block hands the continuation out as its (register-sized) value.
        if resolution.is_none() {
            rust!(self.out, "let {}goto = {{", self.prefix);
        } else {
            rust!(self.out, "{{");
        }

        // take the shifted terminal as the top of the handle, bound as a
        // raw spanned value (it never touches the stack and is never
        // wrapped in the Symbol enum)
        let top = format!("{}sym{}", self.prefix, len - 1);
        let (pattern, content) = self.terminal_pattern_and_content(terminal);
        rust!(
            self.out,
            "let {} = match {}parser.{}lookahead.take() {{",
            top,
            self.prefix,
            self.prefix
        );
        rust!(
            self.out,
            "Some(({}loc1, {}, {}loc2)) => ({}loc1, {}, {}loc2),",
            self.prefix,
            pattern,
            self.prefix,
            self.prefix,
            content,
            self.prefix
        );
        rust!(self.out, "_ => unreachable!(),");
        rust!(self.out, "}};");

        // advance past the shifted terminal
        self.emit_advance_lookahead()?;

        // the target state's ACTION dispatch: the reduction on its
        // lookaheads, an error on anything else
        let target_state = &self.states[target_index.0];
        let mut tokens: Vec<Token> = vec![];
        for (token_set, _) in &target_state.reductions {
            for token in token_set.iter() {
                if !tokens.contains(&token) {
                    tokens.push(token);
                }
            }
        }

        rust!(
            self.out,
            "match &{}parser.{}lookahead {{",
            self.prefix,
            self.prefix
        );
        for (index, token) in tokens.iter().enumerate() {
            let pattern = match *token {
                Token::Terminal(ref s) => format!("Some({})", self.match_terminal_pattern(s)),
                Token::Error => {
                    panic!("Error recovery is not implemented for tail call parsers")
                }
                Token::Eof => "None".to_string(),
            };
            if index < tokens.len() - 1 {
                rust!(self.out, "{} |", pattern);
            } else {
                rust!(self.out, "{} => {{", pattern);
            }
        }

        self.emit_pop_handle(production, resolution.is_none(), true)?;
        self.emit_action_call(production)?;
        self.emit_reduce_push(production, resolution)?;
        if resolution.is_none() {
            rust!(self.out, "{}goto", self.prefix);
        }

        rust!(self.out, "}}"); // reduce arm

        rust!(self.out, "_ => {{");
        self.emit_error(target_state, Some(format!("{}.2.clone()", top)))?;
        rust!(self.out, "}}"); // error arm
        rust!(self.out, "}}"); // match lookahead

        // close the inner block and transition with nothing live
        if resolution.is_none() {
            rust!(self.out, "}};");
        } else {
            rust!(self.out, "}}");
        }
        self.emit_reduce_transition(production, resolution)?;
        Ok(())
    }

    /// Writes the goto function for a state: dispatches on the nonterminal
    /// that was just reduced (communicated through `parser.reduced_nt`; the
    /// reduced symbol itself has already been pushed, paired with this very
    /// function) and enters the target state.
    fn write_goto_fn(&mut self, this_index: StateIndex) -> io::Result<()> {
        let this_state = &self.states[this_index.0];

        rust!(self.out, "");
        if Tls::session().emit_comments {
            rust!(self.out, "// Goto function for state {}", this_index.0);
        }
        // A goto function may be unreferenced (if its state never pushes a
        // symbol) or unreachable-but-referenced (a state that is never the
        // survivor of a reduction still names its goto row when pushing).
        rust!(self.out, "#[allow(dead_code)]");
        self.emit_parser_fn_header(format!("{}goto{}", self.prefix, this_index.0))?;

        if this_state.gotos.is_empty() {
            // this state is never exposed by a reduction
            rust!(self.out, "unreachable!()");
        } else {
            rust!(
                self.out,
                "match {}parser.{}reduced_nt {{",
                self.prefix,
                self.prefix
            );
            for (nt, &next_index) in &this_state.gotos {
                let index = self.nonterminal_index(nt);
                rust!(self.out, "{} => {{", index);
                rust!(
                    self.out,
                    "{}transition!({}state{}({}{}parser))",
                    self.prefix,
                    self.prefix,
                    next_index.0,
                    self.grammar.user_parameter_refs(),
                    self.prefix
                );
                rust!(self.out, "}}");
            }
            rust!(self.out, "_ => unreachable!(),");
            rust!(self.out, "}}");
        }

        rust!(self.out, "}}"); // fn
        Ok(())
    }

    /// Writes the shared reduce functions for (non-empty, non-start)
    /// productions. Unlike classic recursive ascent -- where the reduce code
    /// is specialized to each state because the handle lives in the state
    /// functions' frames -- the handle here is always on the explicit stack,
    /// so the reduce code depends only on the production plus, when the
    /// surviving state can be statically resolved, on that survivor; one
    /// function is emitted per (production, resolution) pair that some state
    /// actually transitions to.
    fn write_reduce_fns(&mut self) -> io::Result<()> {
        let mut seen: Vec<(usize, Option<usize>)> = vec![];
        let mut variants: Vec<(&'grammar Production, Option<StateIndex>)> = vec![];
        for (index, state) in self.states.iter().enumerate() {
            for &(_, production) in &state.reductions {
                if production.nonterminal == self.start_symbol || production.symbols.is_empty() {
                    continue;
                }
                let resolution = self.reduce_resolution(StateIndex(index), &production.symbols);
                let key = (
                    self.custom.reduce_indices[production],
                    resolution.map(|s| s.0),
                );
                if !seen.contains(&key) {
                    seen.push(key);
                    variants.push((production, resolution));
                }
            }
        }
        for (production, resolution) in variants {
            self.write_reduce_fn(production, resolution)?;
        }
        Ok(())
    }

    fn write_reduce_fn(
        &mut self,
        production: &'grammar Production,
        resolution: Option<StateIndex>,
    ) -> io::Result<()> {
        rust!(self.out, "");
        rust!(self.out, "// {:?}", production);
        if let Some(survivor) = resolution {
            rust!(
                self.out,
                "// (specialized for reduce sites whose survivor is statically state {})",
                survivor.0
            );
        }
        // some productions may not be reachable from the start symbol
        rust!(self.out, "#[allow(dead_code)]");
        let name = self.reduce_fn_name(production, resolution);
        self.emit_parser_fn_header(name)?;

        // The pop-action-push sequence runs inside an inner block so all its
        // stack slots are dead before the transition; for a dynamic survivor
        // the block hands out the (register-sized) continuation. See the
        // dispatch comment in `write_state_fn`.
        if resolution.is_none() {
            rust!(self.out, "let {}goto = {{", self.prefix);
        } else {
            rust!(self.out, "{{");
        }
        self.emit_pop_handle(production, resolution.is_none(), false)?;
        self.emit_action_call(production)?;
        self.emit_reduce_push(production, resolution)?;
        if resolution.is_none() {
            rust!(self.out, "{}goto", self.prefix);
            rust!(self.out, "}};");
        } else {
            rust!(self.out, "}}");
        }
        self.emit_reduce_transition(production, resolution)?;

        rust!(self.out, "}}"); // fn
        Ok(())
    }

    /// Pushes the reduced nonterminal, paired with the surviving state's
    /// goto row. When the survivor was statically resolved, the pushed
    /// continuation is a compile-time constant; otherwise it is the
    /// continuation popped from the deepest handle entry (bound as `goto`),
    /// and the reduced-nonterminal discriminant is stored for the dynamic
    /// goto dispatch.
    fn emit_reduce_push(
        &mut self,
        production: &'grammar Production,
        resolution: Option<StateIndex>,
    ) -> io::Result<()> {
        let variant_name =
            self.variant_name_for_symbol(&Symbol::Nonterminal(production.nonterminal.clone()));
        match resolution {
            Some(survivor) => {
                rust!(
                    self.out,
                    "{}parser.{}stack.push((({}start, {}Symbol::{}({}nt), {}end), {}Goto({}goto{})));",
                    self.prefix,
                    self.prefix,
                    self.prefix,
                    self.prefix,
                    variant_name,
                    self.prefix,
                    self.prefix,
                    self.prefix,
                    self.prefix,
                    survivor.0
                );
            }
            None => {
                rust!(
                    self.out,
                    "{}parser.{}stack.push((({}start, {}Symbol::{}({}nt), {}end), {}goto));",
                    self.prefix,
                    self.prefix,
                    self.prefix,
                    self.prefix,
                    variant_name,
                    self.prefix,
                    self.prefix,
                    self.prefix
                );
                rust!(
                    self.out,
                    "{}parser.{}reduced_nt = {};",
                    self.prefix,
                    self.prefix,
                    self.nonterminal_index(&production.nonterminal)
                );
            }
        }
        Ok(())
    }

    /// Transfers control to the surviving state's goto row: a direct call
    /// into the goto's target state when the survivor is statically known,
    /// or a dispatch through the popped continuation otherwise.
    fn emit_reduce_transition(
        &mut self,
        production: &'grammar Production,
        resolution: Option<StateIndex>,
    ) -> io::Result<()> {
        match resolution {
            Some(survivor) => {
                let next_index = self.states[survivor.0].gotos[&production.nonterminal];
                rust!(
                    self.out,
                    "{}transition!({}state{}({}{}parser))",
                    self.prefix,
                    self.prefix,
                    next_index.0,
                    self.grammar.user_parameter_refs(),
                    self.prefix
                );
            }
            None => {
                rust!(
                    self.out,
                    "{}transition!(({}goto.0)({}{}parser))",
                    self.prefix,
                    self.prefix,
                    self.grammar.user_parameter_refs(),
                    self.prefix
                );
            }
        }
        Ok(())
    }

    /// Emits the accept code for the start production `S' = S`: pop the
    /// handle, run the action, and return the parsed value (or report an
    /// `ExtraToken` error if there is remaining input).
    fn emit_accept(&mut self, production: &'grammar Production) -> io::Result<()> {
        self.emit_pop_handle(production, false, false)?;
        self.emit_action_call(production)?;
        rust!(
            self.out,
            "{}parser.{}result = Some(match {}parser.{}lookahead.take() {{",
            self.prefix,
            self.prefix,
            self.prefix,
            self.prefix
        );
        rust!(self.out, "None => Ok({}nt),", self.prefix);
        rust!(
            self.out,
            "Some({}token) => Err({}lalrpop_util::ParseError::ExtraToken {{ token: {}token }}),",
            self.prefix,
            self.prefix,
            self.prefix
        );
        rust!(self.out, "}});");
        rust!(self.out, "return;");
        Ok(())
    }

    /// Emits the reduce code for an empty production directly inside the
    /// state function: nothing is popped, so the surviving state is the
    /// current state itself and the goto is a direct, static transition.
    fn emit_empty_reduce(
        &mut self,
        this_index: StateIndex,
        production: &'grammar Production,
    ) -> io::Result<()> {
        let this_state = &self.states[this_index.0];
        let next_index = this_state.gotos[&production.nonterminal];

        // inner block so all stack slots are dead before the transition;
        // see the dispatch comment in `write_state_fn`
        rust!(self.out, "{{");

        // the span of an empty production is empty; anchor it at the
        // lookahead, or failing that at the end of the top stack symbol
        rust!(
            self.out,
            "let {}start = match &{}parser.{}lookahead {{",
            self.prefix,
            self.prefix,
            self.prefix
        );
        rust!(
            self.out,
            "Some({}t) => {}t.0.clone(),",
            self.prefix,
            self.prefix
        );
        rust!(
            self.out,
            "None => match {}parser.{}stack.last() {{",
            self.prefix,
            self.prefix
        );
        rust!(
            self.out,
            "Some({}entry) => ({}entry.0).2.clone(),",
            self.prefix,
            self.prefix
        );
        rust!(self.out, "None => Default::default(),");
        rust!(self.out, "}},");
        rust!(self.out, "}};");
        rust!(
            self.out,
            "let {}end = {}start.clone();",
            self.prefix,
            self.prefix
        );

        self.emit_action_call(production)?;

        let variant_name =
            self.variant_name_for_symbol(&Symbol::Nonterminal(production.nonterminal.clone()));
        rust!(
            self.out,
            "{}parser.{}stack.push((({}start, {}Symbol::{}({}nt), {}end), {}Goto({}goto{})));",
            self.prefix,
            self.prefix,
            self.prefix,
            self.prefix,
            variant_name,
            self.prefix,
            self.prefix,
            self.prefix,
            self.prefix,
            this_index.0
        );
        rust!(self.out, "}}"); // inner block
        rust!(
            self.out,
            "{}transition!({}state{}({}{}parser))",
            self.prefix,
            self.prefix,
            next_index.0,
            self.grammar.user_parameter_refs(),
            self.prefix
        );
        Ok(())
    }

    /// Emits code that pops the production's handle off the explicit stack,
    /// binding each `symN` to a spanned value of its true type and computing
    /// the `start`/`end` locations. If `bind_goto` is true, the continuation
    /// found on the deepest entry is bound as `goto` (a statically-resolved
    /// reduce does not need it). If `top_is_local` is true, the topmost
    /// handle symbol is not popped: the caller has already bound it as
    /// `sym{N-1}` (the fused shift-reduce case).
    ///
    /// Popping and downcasting happen together inside the per-variant
    /// `pop_VariantN` helper functions rather than inline: the helpers keep
    /// the type-mismatch panic (and the unwind cleanups of the symbols still
    /// held across it) out of the reduce function's own MIR. Inlining the
    /// downcasts here makes rustc guard those cleanups with drop flags,
    /// which gives the final transition an unwind edge -- and a call with an
    /// unwind edge can never become a tail call.
    fn emit_pop_handle(
        &mut self,
        production: &'grammar Production,
        bind_goto: bool,
        top_is_local: bool,
    ) -> io::Result<()> {
        let len = production.symbols.len();
        assert!(len > 0);
        let pop_count = if top_is_local { len - 1 } else { len };
        assert!(!(bind_goto && pop_count == 0));

        if pop_count > 1 {
            // By asserting that there are enough elements to pop before
            // popping multiple elements we may help LLVM to optimize better
            // since it does not need to generate panic branches for each
            // unwrap
            rust!(
                self.out,
                "assert!({}parser.{}stack.len() >= {});",
                self.prefix,
                self.prefix,
                pop_count
            );
        }

        // pop the handle, top of the stack first; the deepest entry carries
        // the goto row of the surviving state
        for index in (0..pop_count).rev() {
            let goto = if index == 0 && bind_goto {
                format!("{}goto", self.prefix)
            } else {
                "_".to_string()
            };
            let variant_name = self.variant_name_for_symbol(&production.symbols[index]);
            rust!(
                self.out,
                "let ({}sym{}, {}) = {}pop_{}(&mut {}parser.{}stack);",
                self.prefix,
                index,
                goto,
                self.prefix,
                variant_name,
                self.prefix,
                self.prefix
            );
        }

        rust!(
            self.out,
            "let {}start = {}sym0.0.clone();",
            self.prefix,
            self.prefix
        );
        rust!(
            self.out,
            "let {}end = {}sym{}.2.clone();",
            self.prefix,
            self.prefix,
            len - 1
        );
        Ok(())
    }

    /// One `#[inline(never)]` wrapper around each production's action code.
    /// The parser functions call the action through these shims rather than
    /// directly, so that user action code is never inlined into a parser
    /// function: action code can contain constructs (for example, a panic
    /// path that passes a local by reference) that make LLVM consider the
    /// surrounding function's stack as escaping, which stops the final
    /// transition from being compiled as a tail call.
    fn write_action_shims(&mut self) -> io::Result<()> {
        let productions: Vec<&'grammar Production> = self
            .grammar
            .nonterminals
            .values()
            .flat_map(|nt| &nt.productions)
            .collect();
        for production in productions {
            let index = self.custom.reduce_indices[production];
            let loc_type = self.types.terminal_loc_type();

            let parameters: Vec<String> = if production.symbols.is_empty() {
                vec![
                    format!("{}start: &{}", self.prefix, loc_type),
                    format!("{}end: &{}", self.prefix, loc_type),
                ]
            } else {
                production
                    .symbols
                    .iter()
                    .enumerate()
                    .map(|(i, symbol)| {
                        format!(
                            "{}sym{}: {}",
                            self.prefix,
                            i,
                            self.types.spanned_type(symbol.ty(self.types).clone())
                        )
                    })
                    .collect()
            };

            let nt_type = self.types.nonterminal_type(&production.nonterminal);
            let return_type = if self.grammar.action_is_fallible(production.action) {
                format!("Result<{}, {}>", nt_type, self.types.parse_error_type())
            } else {
                format!("{}", nt_type)
            };

            let mut args: Vec<String> = (0..production.symbols.len())
                .map(|i| format!("{}sym{}", self.prefix, i))
                .collect();
            if args.is_empty() {
                args.push(format!("{}start", self.prefix));
                args.push(format!("{}end", self.prefix));
            }

            rust!(self.out, "#[allow(dead_code)]");
            rust!(self.out, "#[inline(never)]");
            self.out
                .fn_header(
                    &Visibility::Priv,
                    format!("{}call_action{}", self.prefix, index),
                )
                .with_grammar(self.grammar)
                .with_parameters(parameters)
                .with_return_type(return_type)
                .emit()?;
            rust!(self.out, "{{");
            rust!(
                self.out,
                "{}::{}action{}::<{}>({}{})",
                self.action_module,
                self.prefix,
                production.action.index(),
                Sep(", ", &self.grammar.non_lifetime_type_parameters()),
                self.grammar.user_parameter_refs(),
                Sep(", ", &args)
            );
            rust!(self.out, "}}");
            rust!(self.out, "");
        }
        Ok(())
    }

    /// Emits the call to the action code, binding the result as `nt`. For
    /// non-empty productions the `symN` triples must already be in scope;
    /// for empty productions, `start`/`end` locations must be.
    fn emit_action_call(&mut self, production: &'grammar Production) -> io::Result<()> {
        let index = self.custom.reduce_indices[production];
        let mut args: Vec<String> = (0..production.symbols.len())
            .map(|i| format!("{}sym{}", self.prefix, i))
            .collect();
        if args.is_empty() {
            args.push(format!("&{}start", self.prefix));
            args.push(format!("&{}end", self.prefix));
        }

        let is_fallible = self.grammar.action_is_fallible(production.action);
        if is_fallible {
            rust!(
                self.out,
                "let {}nt = match {}call_action{}::<{}>({}{}) {{",
                self.prefix,
                self.prefix,
                index,
                Sep(", ", &self.grammar.non_lifetime_type_parameters()),
                self.grammar.user_parameter_refs(),
                Sep(", ", &args)
            );
            rust!(self.out, "Ok({}v) => {}v,", self.prefix, self.prefix);
            rust!(
                self.out,
                "Err({}e) => {{ {}parser.{}result = Some(Err({}e)); return; }}",
                self.prefix,
                self.prefix,
                self.prefix,
                self.prefix
            );
            rust!(self.out, "}};");
        } else {
            rust!(
                self.out,
                "let {}nt = {}call_action{}::<{}>({}{});",
                self.prefix,
                self.prefix,
                index,
                Sep(", ", &self.grammar.non_lifetime_type_parameters()),
                self.grammar.user_parameter_refs(),
                Sep(", ", &args)
            );
        }
        Ok(())
    }

    /// Emits the header of a state/goto/reduce function. These all share the
    /// exact same signature: the grammar's user parameters and the parser
    /// struct. This uniformity is required for `become` (which insists that
    /// caller and callee signatures match) and makes every transition a
    /// sibling call for LLVM otherwise.
    ///
    /// The functions are `#[inline(never)]`: when LLVM inlines one parser
    /// function into another, the inlinee's tail calls land in the middle of
    /// the merged function (followed by the inliner's block structure and
    /// stack-slot cleanup), which stops the backend from emitting them as
    /// jumps -- and the native stack starts growing per transition again.
    /// Keeping each function standalone keeps its transitions in genuine
    /// tail position.
    fn emit_parser_fn_header(&mut self, name: String) -> io::Result<()> {
        let tokens_bound = self.tokens_bound();
        let parameters = vec![format!(
            "{}parser: &mut {}Parser<{}>",
            self.prefix,
            self.prefix,
            self.parser_type_args()
        )];
        rust!(self.out, "#[inline(never)]");
        self.out
            .fn_header(&Visibility::Priv, name)
            .with_grammar(self.grammar)
            .with_type_parameters(Some(tokens_bound))
            .with_parameters(parameters)
            .emit()?;
        rust!(self.out, "{{");
        Ok(())
    }

    /// Emit a pattern that matches `id` but doesn't extract any data.
    fn match_terminal_pattern(&mut self, id: &TerminalString) -> String {
        let pattern = self.grammar.pattern(id).map(&mut |_| "_");
        format!("(_, {pattern}, _)")
    }

    /// A pattern binding the data of terminal `id`, and the expression
    /// rebuilding the terminal's value from those bindings.
    fn terminal_pattern_and_content(&mut self, id: &TerminalString) -> (String, String) {
        let mut pattern_names = vec![];
        let pattern = self.grammar.pattern(id).map(&mut |_| {
            let index = pattern_names.len();
            pattern_names.push(format!("{}tok{}", self.prefix, index));
            pattern_names.last().cloned().unwrap()
        });

        let mut pattern = format!("{pattern}");
        let content = if pattern_names.is_empty() {
            // no data extracted: the value is the token itself
            pattern = format!("{}tok @ {}", self.prefix, pattern);
            format!("{}tok", self.prefix)
        } else if pattern_names.len() == 1 {
            pattern_names.pop().unwrap()
        } else {
            format!("({})", pattern_names.join(", "))
        };
        (pattern, content)
    }

    /// Statically resolves which state survives popping `prefix` symbols
    /// while in state `from`: the survivor must be a state from which
    /// pushing `prefix` reaches `from`. If exactly one state qualifies, the
    /// continuation of a reduction popping that prefix is a compile-time
    /// constant, and both the dispatch through the stored goto pointer and
    /// the goto's own dispatch on the reduced nonterminal collapse into a
    /// direct transition. This is the generator-level analogue of the
    /// devirtualization that tc_args obtains from LLVM constant propagation
    /// (report section 3.5): the LR invariant "the continuation below this
    /// handle is that state's goto row" is restated where it is statically
    /// visible.
    fn reduce_resolution(&self, from: StateIndex, prefix: &[Symbol]) -> Option<StateIndex> {
        let survivors = self.custom.graph.trace_back(from, prefix);
        match survivors[..] {
            [survivor] => Some(survivor),
            _ => None,
        }
    }

    /// If `target` is a *reduce-only* state -- no shifts, no gotos, and one
    /// single reduced production -- then a shift into it can be fused: the
    /// target state would only consult one lookahead token and pop a handle
    /// whose top we are about to push. Returns that production.
    fn fused_production(&self, target: StateIndex) -> Option<&'grammar Production> {
        let state = &self.states[target.0];
        if !state.shifts.is_empty() || !state.gotos.is_empty() {
            return None;
        }
        let mut productions = state.reductions.iter().map(|(_, p)| *p);
        let first = productions.next()?;
        if productions.all(|p| p == first)
            && first.nonterminal != self.start_symbol
            && !first.symbols.is_empty()
        {
            Some(first)
        } else {
            None
        }
    }

    /// The name of the shared reduce fn for a production, specialized by the
    /// statically-resolved survivor when there is one.
    fn reduce_fn_name(
        &self,
        production: &'grammar Production,
        resolution: Option<StateIndex>,
    ) -> String {
        let index = self.custom.reduce_indices[production];
        match resolution {
            Some(survivor) => format!("{}reduce{}via{}", self.prefix, index, survivor.0),
            None => format!("{}reduce{}", self.prefix, index),
        }
    }

    fn variant_name_for_symbol(&self, s: &Symbol) -> String {
        self.custom.variant_names[s].clone()
    }

    fn nonterminal_index(&self, nt: &NonterminalString) -> usize {
        self.custom
            .all_nonterminals
            .iter()
            .position(|x| x == nt)
            .unwrap()
    }

    /// The declaration list for the parser/goto types and functions: the
    /// grammar's type parameters followed by the token iterator.
    fn parser_type_decls(&self) -> String {
        let mut decls: Vec<String> = self
            .grammar
            .type_parameters
            .iter()
            .map(|tp| tp.to_string())
            .collect();
        decls.push(self.tokens_bound());
        decls.join(", ")
    }

    /// Same as `parser_type_decls`, but as an argument list (no bounds).
    fn parser_type_args(&self) -> String {
        let mut args: Vec<String> = self
            .grammar
            .type_parameters
            .iter()
            .map(|tp| tp.to_string())
            .collect();
        args.push(format!("{}TOKENS", self.prefix));
        args.join(", ")
    }

    /// Where clauses for the parser/goto types: the grammar's, if any. (The
    /// token iterator bound lives in the declaration list.)
    fn parser_where_clauses(&self) -> String {
        let mut clauses: Vec<String> = self
            .grammar
            .where_clauses
            .iter()
            .map(|wc| wc.to_string())
            .collect();
        clauses.push(self.tokens_bound());
        clauses.join(", ")
    }

    fn tokens_bound(&self) -> String {
        format!(
            "{}TOKENS: Iterator<Item = Result<{}, {}>>",
            self.prefix,
            self.triple_type(),
            self.types.parse_error_type()
        )
    }

    /// The parameter types of the unified fn signature, for the fn pointer
    /// type inside `Goto`.
    fn unified_fn_params(&self) -> String {
        let mut params: Vec<String> = self
            .grammar
            .parameters
            .iter()
            .map(|p| p.ty.to_string())
            .collect();
        params.push(format!(
            "&mut {}Parser<{}>",
            self.prefix,
            self.parser_type_args()
        ));
        params.join(", ")
    }

    fn result_type(&self) -> String {
        format!(
            "Result<{}, {}>",
            self.types.nonterminal_type(&self.start_symbol),
            self.types.parse_error_type()
        )
    }

    fn spanned_symbol_type(&self) -> String {
        let loc_type = self.types.terminal_loc_type();
        format!(
            "({}, {}Symbol<{}>, {})",
            loc_type,
            self.prefix,
            Sep(", ", &self.custom.symbol_type_params),
            loc_type
        )
    }

    fn triple_type(&self) -> TypeRepr {
        self.types.triple_type()
    }
}
