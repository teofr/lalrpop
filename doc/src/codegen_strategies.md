# Code generation strategies

LALRPOP can compile the same LR(1)/LALR(1) automaton into several different
styles of Rust code. The strategy is selected with an attribute on the
`grammar` declaration:

```lalrpop
#[tail_call]
grammar;
```

The available strategies are:

- `#[table_driven]` *(default)* — the classic formulation: the automaton is
  encoded as ACTION/GOTO tables, and a driver loop interprets them, keeping
  an explicit stack of states and values. This is the most battle-tested
  backend and the only one supporting error recovery (the `!` symbol).

- `#[recursive_ascent]` — each LR state becomes a Rust function, and the
  automaton's stack *is* the native call stack: a shift is a function call,
  and a reduction returns through as many frames as the production has
  symbols. Dispatch overhead is replaced by direct control flow, but the
  native stack grows with the recursion depth of your grammar over the
  input, so deeply nested input can overflow the stack. Error recovery is
  not supported.

- `#[tail_call]` — a hybrid of the two, sometimes called *recursive ascent
  pushdown*: each LR state is still its own function, but the parse stack is
  an explicit `Vec` of `(value, continuation)` pairs, where the continuation
  is the function pointer implementing the GOTO row of the state below.
  Every transition — shift, reduce, goto — is a call in tail position with
  an identical signature, so the parser needs only *constant* native stack
  no matter how large or deeply nested the input is, while keeping the
  directly-coded control flow of recursive ascent. Error recovery is not
  supported.

- `#[test_all]` — a testing harness that generates all of the above and
  asserts they produce identical results. Used by LALRPOP's own test suite;
  not intended for external consumption.

## `#[tail_call]` and stack usage

Rust does not (yet) guarantee tail-call elimination, so what "constant
native stack" means for a `#[tail_call]` parser depends on how it is
compiled:

- **Optimized builds** (`opt-level` 2 or 3): all transitions share one exact
  signature with register-sized arguments and return values, precisely so
  that LLVM's sibling-call optimization applies. In practice every
  transition compiles to a jump and the parser runs in O(1) native stack.
  This is reliable but is a property of the optimizer, not a language
  guarantee.

- **Debug builds**: no tail-call elimination is performed, and the native
  stack grows with every *token* (not just with nesting depth). Large
  inputs will overflow the stack in unoptimized builds.

- **Guaranteed elimination with `become`** (nightly): the generated code
  contains both plain-`return` transitions and `become` transitions (RFC
  3407, feature `explicit_tail_calls`), selected by a `cfg` flag. If you
  build with

  ```text
  RUSTFLAGS="--cfg lalrpop_tail_call_become"
  ```

  and enable the feature at your crate root:

  ```rust
  #![cfg_attr(lalrpop_tail_call_become, feature(explicit_tail_calls))]
  #![cfg_attr(lalrpop_tail_call_become, allow(incomplete_features))]
  ```

  then every transition uses `become` and the O(1)-native-stack property is
  enforced by the compiler in every profile, including debug builds. This
  requires a nightly toolchain.

One further caveat: grammar parameters (`grammar(x: Foo)`) are passed by
value through every transition. Keep them register-sized — references,
integers, `Copy` types — or they may defeat the sibling-call optimization
that the optimized non-`become` build relies on.

## Which one should I use?

Use the default table-driven backend unless you have a reason not to: it is
the most widely used and supports error recovery. Reach for `#[tail_call]`
when parser throughput matters or when you must parse adversarially deep
input without overflowing the stack in optimized builds; prefer it over
`#[recursive_ascent]`, which it strictly improves upon in stack behavior.
As always with performance, measure on your own grammar and inputs — the
repository's `lalrpop-test/benches` harness shows how to set up such a
comparison.
