#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![warn(clippy::std_instead_of_core)]
#![warn(clippy::std_instead_of_alloc)]
#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

extern crate alloc;

use alloc::string::ToString;
use alloc::{vec, vec::Vec};
use core::fmt::Debug;
use facet_core::{NumericType, PrimitiveType};

mod error;
use alloc::borrow::Cow;

pub use error::*;

mod span;
use facet_core::{Characteristic, Def, Facet, FieldFlags, PointerType, StructKind, Type, UserType};
use owo_colors::OwoColorize;
pub use span::*;

use facet_reflect::{HeapValue, Partial, ReflectError};
use log::trace;

#[derive(PartialEq, Debug, Clone)]
/// A scalar value used during deserialization.
/// `u64` and `i64` are separated because `i64` doesn't fit in `u64`,
/// but having `u64` is a fast path for 64-bit architectures — no need to
/// go through `u128` / `i128` for everything
pub enum Scalar<'input> {
    /// Owned or borrowed string data.
    String(Cow<'input, str>),
    /// Unsigned 64-bit integer scalar.
    U64(u64),
    /// Signed 64-bit integer scalar.
    I64(i64),
    /// 64-bit floating-point scalar.
    F64(f64),
    /// 128-bit unsigned integer scalar.
    U128(u128),
    /// 128-bit signed integer scalar.
    I128(i128),
    /// Boolean scalar.
    Bool(bool),
    /// Null scalar (e.g. for formats supporting explicit null).
    Null,
}

#[derive(PartialEq, Debug, Clone)]
/// Expected next input token or structure during deserialization.
pub enum Expectation {
    /// Accept a value.
    Value,
    /// Expect an object key or the end of an object.
    ObjectKeyOrObjectClose,
    /// Expect a value inside an object.
    ObjectVal,
    /// Expect a list item or the end of a list.
    ListItemOrListClose,
}

#[derive(PartialEq, Debug, Clone)]
/// Outcome of parsing the next input element.
pub enum Outcome<'input> {
    /// Parsed a scalar value.
    Scalar(Scalar<'input>),
    /// Starting a list/array.
    ListStarted,
    /// Ending a list/array.
    ListEnded,
    /// Starting an object/map.
    ObjectStarted,
    /// Ending an object/map.
    ObjectEnded,
}

impl<'input> From<Scalar<'input>> for Outcome<'input> {
    fn from(scalar: Scalar<'input>) -> Self {
        Outcome::Scalar(scalar)
    }
}

use core::fmt;

/// Display implementation for `Outcome`, focusing on user-friendly descriptions.
impl fmt::Display for Outcome<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Outcome::Scalar(scalar) => write!(f, "scalar {scalar}"),
            Outcome::ListStarted => write!(f, "list start"),
            Outcome::ListEnded => write!(f, "list end"),
            Outcome::ObjectStarted => write!(f, "object start"),
            Outcome::ObjectEnded => write!(f, "object end"),
        }
    }
}

/// Display implementation for `Scalar`, for use in displaying `Outcome`.
impl fmt::Display for Scalar<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scalar::String(s) => write!(f, "string \"{s}\""),
            Scalar::U64(val) => write!(f, "u64 {val}"),
            Scalar::I64(val) => write!(f, "i64 {val}"),
            Scalar::F64(val) => write!(f, "f64 {val}"),
            Scalar::U128(val) => write!(f, "u128 {val}"),
            Scalar::I128(val) => write!(f, "i128 {val}"),
            Scalar::Bool(val) => write!(f, "bool {val}"),
            Scalar::Null => write!(f, "null"),
        }
    }
}

impl Outcome<'_> {
    fn into_owned(self) -> Outcome<'static> {
        match self {
            Outcome::Scalar(scalar) => {
                let owned_scalar = match scalar {
                    Scalar::String(cow) => Scalar::String(Cow::Owned(cow.into_owned())),
                    Scalar::U64(val) => Scalar::U64(val),
                    Scalar::I64(val) => Scalar::I64(val),
                    Scalar::F64(val) => Scalar::F64(val),
                    Scalar::U128(val) => Scalar::U128(val),
                    Scalar::I128(val) => Scalar::I128(val),
                    Scalar::Bool(val) => Scalar::Bool(val),
                    Scalar::Null => Scalar::Null,
                };
                Outcome::Scalar(owned_scalar)
            }
            Outcome::ListStarted => Outcome::ListStarted,
            Outcome::ListEnded => Outcome::ListEnded,
            Outcome::ObjectStarted => Outcome::ObjectStarted,
            Outcome::ObjectEnded => Outcome::ObjectEnded,
        }
    }
}

/// Carries the current parsing state and the in-progress value during deserialization.
/// This bundles the mutable context that must be threaded through parsing steps.
pub struct NextData<'input, 'facet>
where
    'input: 'facet,
{
    /// The offset we're supposed to start parsing from
    start: usize,

    /// Controls the parsing flow and stack state.
    runner: StackRunner<'input>,

    /// Holds the intermediate representation of the value being built.
    pub wip: Partial<'facet>,
}

impl<'input, 'facet> NextData<'input, 'facet>
where
    'input: 'facet,
{
    /// Returns the input (from the start! not from the current position)
    pub fn input(&self) -> &'input [u8] {
        self.runner.input
    }

    /// Returns the parsing start offset.
    pub fn start(&self) -> usize {
        self.start
    }
}

/// The result of advancing the parser: updated state and parse outcome or error.
pub type NextResult<'input, 'facet, T, E> = (NextData<'input, 'facet>, Result<T, E>);

/// Trait defining a deserialization format.
/// Provides the next parsing step based on current state and expected input.
pub trait Format {
    /// The lowercase source ID of the format, used for error reporting.
    fn source(&self) -> &'static str;

    /// Advance the parser with current state and expectation, producing the next outcome or error.
    #[allow(clippy::type_complexity)]
    fn next<'input, 'facet>(
        &mut self,
        nd: NextData<'input, 'facet>,
        expectation: Expectation,
    ) -> NextResult<'input, 'facet, Spanned<Outcome<'input>>, Spanned<DeserErrorKind>>;

    /// Skip the next value; used to ignore an input.
    #[allow(clippy::type_complexity)]
    fn skip<'input, 'facet>(
        &mut self,
        nd: NextData<'input, 'facet>,
    ) -> NextResult<'input, 'facet, Span, Spanned<DeserErrorKind>>;
}

/// Instructions guiding the parsing flow, indicating the next expected action or token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Instruction {
    /// Expect a value, specifying the context or reason.
    Value(ValueReason),
    /// Skip the next value; used to ignore an input.
    SkipValue,
    /// Indicate completion of a structure or value; triggers popping from stack.
    Pop(PopReason),
    /// Expect an object key or the end of an object.
    ObjectKeyOrObjectClose,
    /// Expect a list item or the end of a list.
    ListItemOrListClose,
}

/// Reasons for expecting a value, reflecting the current parse context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueReason {
    /// Parsing at the root level.
    TopLevel,
    /// Parsing a value inside an object.
    ObjectVal,
}

/// Reasons for popping a state from the stack, indicating why a scope is ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PopReason {
    /// Ending the top-level parsing scope.
    TopLevel,
    /// Ending a value within an object.
    ObjectVal,
    /// Ending value within a list
    ListVal,
    /// Ending a `Some()` in an option
    Some,
    /// Ending a smart pointer (ie. wrapping a `T` back into a `Box<T>`, or `Arc<T>` etc.)
    SmartPointer,
    /// Ending a wrapper value such as a newtype
    Wrapper,
}

mod deser_impl {
    use super::*;

    /// Deserialize a value of type `T` from raw input bytes using format `F`.
    ///
    /// This function sets up the initial working state and drives the deserialization process,
    /// ensuring that the resulting value is fully materialized and valid.
    pub fn deserialize<'input, 'facet, T, F>(
        input: &'input [u8],
        format: &mut F,
    ) -> Result<T, DeserError<'input>>
    where
        T: Facet<'facet>,
        F: Format,
        'input: 'facet,
    {
        // Run the entire deserialization process and capture any errors
        let result: Result<T, DeserError<'input>> = {
            // Step 1: Allocate shape
            let wip = match Partial::alloc_shape(T::SHAPE) {
                Ok(wip) => wip,
                Err(e) => {
                    let default_span = Span::default();
                    return Err(DeserError::new_reflect(e, input, default_span));
                }
            };

            // Step 2: Run deserialize_wip
            let heap_value = match deserialize_wip(wip, input, format) {
                Ok(val) => val,
                Err(e) => {
                    return Err(e);
                }
            };

            // Step 3: Materialize
            match heap_value.materialize() {
                Ok(val) => Ok(val),
                Err(e) => {
                    let default_span = Span::default();
                    return Err(DeserError::new_reflect(e, input, default_span));
                }
            }
        };

        result
    }
}

/// Deserialize a value of type `T` from raw input bytes using format `F`.
///
/// This function sets up the initial working state and drives the deserialization process,
/// ensuring that the resulting value is fully materialized and valid.
pub fn deserialize<'input, 'facet, T, F>(
    input: &'input [u8],
    format: F,
) -> Result<T, DeserError<'input>>
where
    T: Facet<'facet>,
    F: Format,
    'input: 'facet,
{
    let mut format_copy = format;
    deser_impl::deserialize(input, &mut format_copy)
}

/// Deserializes a working-in-progress value into a fully materialized heap value.
/// This function drives the parsing loop until the entire input is consumed and the value is complete.
pub fn deserialize_wip<'input, 'facet, F>(
    mut wip: Partial<'facet>,
    input: &'input [u8],
    format: &mut F,
) -> Result<HeapValue<'facet>, DeserError<'input>>
where
    F: Format,
    'input: 'facet,
{
    // This struct is just a bundle of the state that we need to pass around all the time.
    let mut runner = StackRunner {
        original_input: input,
        input,
        stack: vec![
            Instruction::Pop(PopReason::TopLevel),
            Instruction::Value(ValueReason::TopLevel),
        ],
        last_span: Span::new(0, 0),
        format_source: format.source(),
        array_indices: Vec::new(),
        enum_tuple_field_count: None,
        enum_tuple_current_field: None,
    };

    macro_rules! next {
        ($runner:ident, $wip:ident, $expectation:expr, $method:ident) => {{
            let nd = NextData {
                start: $runner.last_span.end(), // or supply the appropriate start value if available
                runner: $runner,
                wip: $wip,
            };
            let (nd, res) = format.next(nd, $expectation);
            $runner = nd.runner;
            $wip = nd.wip;
            let outcome = res.map_err(|span_kind| {
                $runner.last_span = span_kind.span;
                $runner.err(span_kind.node)
            })?;
            $runner.last_span = outcome.span;
            $wip = $runner.$method($wip, outcome)?;
        }};
    }

    loop {
        // Note: frames_count() is no longer available in the new Partial API
        // This was used for debugging/assertions only

        let insn = match runner.stack.pop() {
            Some(insn) => insn,
            None => unreachable!("Instruction stack is empty"),
        };

        trace!("Instruction {:?}", insn.bright_red());

        match insn {
            Instruction::Pop(reason) => {
                wip = runner.pop(wip, reason)?;

                if reason == PopReason::TopLevel {
                    // Exit all nested frames (e.g., from flattened fields) before building
                    while wip.frame_count() > 1 {
                        wip.end().map_err(|e| runner.reflect_err(e))?;
                    }
                    return wip.build().map_err(|e| runner.reflect_err(e));
                } else {
                    wip.end().map_err(|e| runner.reflect_err(e))?;
                }
            }
            Instruction::Value(_why) => {
                let expectation = match _why {
                    ValueReason::TopLevel => Expectation::Value,
                    ValueReason::ObjectVal => Expectation::ObjectVal,
                };
                next!(runner, wip, expectation, value);
            }
            Instruction::ObjectKeyOrObjectClose => {
                next!(
                    runner,
                    wip,
                    Expectation::ObjectKeyOrObjectClose,
                    object_key_or_object_close
                );
            }
            Instruction::ListItemOrListClose => {
                next!(
                    runner,
                    wip,
                    Expectation::ListItemOrListClose,
                    list_item_or_list_close
                );
            }
            Instruction::SkipValue => {
                // Call F::skip to skip over the next value in the input
                let nd = NextData {
                    start: runner.last_span.end(),
                    runner,
                    wip,
                };
                let (nd, res) = format.skip(nd);
                runner = nd.runner;
                wip = nd.wip;
                // Only propagate error, don't modify wip, since skip just advances input
                let span = res.map_err(|span_kind| {
                    runner.last_span = span_kind.span;
                    runner.err(span_kind.node)
                })?;
                // do the actual skip
                runner.last_span = span;
            }
        }
    }
}

/// Helper function to check if an f64 has no fractional part
/// This is needed for no-std compatibility where f64::fract() is not available
#[inline]
fn has_no_fractional_part(value: f64) -> bool {
    value == (value as i64) as f64
}

/// Trait for numeric type conversions
trait NumericConvert {
    fn to_i8(&self) -> Option<i8>;
    fn to_i16(&self) -> Option<i16>;
    fn to_i32(&self) -> Option<i32>;
    fn to_i64(&self) -> Option<i64>;
    fn to_i128(&self) -> Option<i128>;
    fn to_isize(&self) -> Option<isize>;

    fn to_u8(&self) -> Option<u8>;
    fn to_u16(&self) -> Option<u16>;
    fn to_u32(&self) -> Option<u32>;
    fn to_u64(&self) -> Option<u64>;
    fn to_u128(&self) -> Option<u128>;
    fn to_usize(&self) -> Option<usize>;

    fn to_f32(&self) -> Option<f32>;
    fn to_f64(&self) -> Option<f64>;
}

impl NumericConvert for u64 {
    fn to_i8(&self) -> Option<i8> {
        (*self).try_into().ok()
    }
    fn to_i16(&self) -> Option<i16> {
        (*self).try_into().ok()
    }
    fn to_i32(&self) -> Option<i32> {
        (*self).try_into().ok()
    }
    fn to_i64(&self) -> Option<i64> {
        (*self).try_into().ok()
    }
    fn to_i128(&self) -> Option<i128> {
        Some(*self as i128)
    }
    fn to_isize(&self) -> Option<isize> {
        (*self).try_into().ok()
    }

    fn to_u8(&self) -> Option<u8> {
        (*self).try_into().ok()
    }
    fn to_u16(&self) -> Option<u16> {
        (*self).try_into().ok()
    }
    fn to_u32(&self) -> Option<u32> {
        (*self).try_into().ok()
    }
    fn to_u64(&self) -> Option<u64> {
        Some(*self)
    }
    fn to_u128(&self) -> Option<u128> {
        Some(*self as u128)
    }
    fn to_usize(&self) -> Option<usize> {
        (*self).try_into().ok()
    }

    fn to_f32(&self) -> Option<f32> {
        Some(*self as f32)
    }
    fn to_f64(&self) -> Option<f64> {
        Some(*self as f64)
    }
}

impl NumericConvert for i64 {
    fn to_i8(&self) -> Option<i8> {
        (*self).try_into().ok()
    }
    fn to_i16(&self) -> Option<i16> {
        (*self).try_into().ok()
    }
    fn to_i32(&self) -> Option<i32> {
        (*self).try_into().ok()
    }
    fn to_i64(&self) -> Option<i64> {
        Some(*self)
    }
    fn to_i128(&self) -> Option<i128> {
        Some(*self as i128)
    }
    fn to_isize(&self) -> Option<isize> {
        (*self).try_into().ok()
    }

    fn to_u8(&self) -> Option<u8> {
        (*self).try_into().ok()
    }
    fn to_u16(&self) -> Option<u16> {
        (*self).try_into().ok()
    }
    fn to_u32(&self) -> Option<u32> {
        (*self).try_into().ok()
    }
    fn to_u64(&self) -> Option<u64> {
        (*self).try_into().ok()
    }
    fn to_u128(&self) -> Option<u128> {
        (*self).try_into().ok()
    }
    fn to_usize(&self) -> Option<usize> {
        (*self).try_into().ok()
    }

    fn to_f32(&self) -> Option<f32> {
        Some(*self as f32)
    }
    fn to_f64(&self) -> Option<f64> {
        Some(*self as f64)
    }
}

impl NumericConvert for f64 {
    fn to_i8(&self) -> Option<i8> {
        if has_no_fractional_part(*self) && *self >= i8::MIN as f64 && *self <= i8::MAX as f64 {
            Some(*self as i8)
        } else {
            None
        }
    }
    fn to_i16(&self) -> Option<i16> {
        if has_no_fractional_part(*self) && *self >= i16::MIN as f64 && *self <= i16::MAX as f64 {
            Some(*self as i16)
        } else {
            None
        }
    }
    fn to_i32(&self) -> Option<i32> {
        if has_no_fractional_part(*self) && *self >= i32::MIN as f64 && *self <= i32::MAX as f64 {
            Some(*self as i32)
        } else {
            None
        }
    }
    fn to_i64(&self) -> Option<i64> {
        if has_no_fractional_part(*self) && *self >= i64::MIN as f64 && *self <= i64::MAX as f64 {
            Some(*self as i64)
        } else {
            None
        }
    }
    fn to_i128(&self) -> Option<i128> {
        if has_no_fractional_part(*self) && *self >= i128::MIN as f64 && *self <= i128::MAX as f64 {
            Some(*self as i128)
        } else {
            None
        }
    }
    fn to_isize(&self) -> Option<isize> {
        if has_no_fractional_part(*self) && *self >= isize::MIN as f64 && *self <= isize::MAX as f64
        {
            Some(*self as isize)
        } else {
            None
        }
    }

    fn to_u8(&self) -> Option<u8> {
        if has_no_fractional_part(*self) && *self >= 0.0 && *self <= u8::MAX as f64 {
            Some(*self as u8)
        } else {
            None
        }
    }
    fn to_u16(&self) -> Option<u16> {
        if has_no_fractional_part(*self) && *self >= 0.0 && *self <= u16::MAX as f64 {
            Some(*self as u16)
        } else {
            None
        }
    }
    fn to_u32(&self) -> Option<u32> {
        if has_no_fractional_part(*self) && *self >= 0.0 && *self <= u32::MAX as f64 {
            Some(*self as u32)
        } else {
            None
        }
    }
    fn to_u64(&self) -> Option<u64> {
        if has_no_fractional_part(*self) && *self >= 0.0 && *self <= u64::MAX as f64 {
            Some(*self as u64)
        } else {
            None
        }
    }
    fn to_u128(&self) -> Option<u128> {
        if has_no_fractional_part(*self) && *self >= 0.0 && *self <= u128::MAX as f64 {
            Some(*self as u128)
        } else {
            None
        }
    }
    fn to_usize(&self) -> Option<usize> {
        if has_no_fractional_part(*self) && *self >= 0.0 && *self <= usize::MAX as f64 {
            Some(*self as usize)
        } else {
            None
        }
    }

    fn to_f32(&self) -> Option<f32> {
        Some(*self as f32)
    }
    fn to_f64(&self) -> Option<f64> {
        Some(*self)
    }
}

impl NumericConvert for u128 {
    fn to_i8(&self) -> Option<i8> {
        (*self).try_into().ok()
    }
    fn to_i16(&self) -> Option<i16> {
        (*self).try_into().ok()
    }
    fn to_i32(&self) -> Option<i32> {
        (*self).try_into().ok()
    }
    fn to_i64(&self) -> Option<i64> {
        (*self).try_into().ok()
    }
    fn to_i128(&self) -> Option<i128> {
        Some(*self as i128)
    }
    fn to_isize(&self) -> Option<isize> {
        (*self).try_into().ok()
    }

    fn to_u8(&self) -> Option<u8> {
        (*self).try_into().ok()
    }
    fn to_u16(&self) -> Option<u16> {
        (*self).try_into().ok()
    }
    fn to_u32(&self) -> Option<u32> {
        (*self).try_into().ok()
    }
    fn to_u64(&self) -> Option<u64> {
        (*self).try_into().ok()
    }
    fn to_u128(&self) -> Option<u128> {
        Some(*self)
    }
    fn to_usize(&self) -> Option<usize> {
        (*self).try_into().ok()
    }

    fn to_f32(&self) -> Option<f32> {
        Some(*self as f32)
    }
    fn to_f64(&self) -> Option<f64> {
        Some(*self as f64)
    }
}

impl NumericConvert for i128 {
    fn to_i8(&self) -> Option<i8> {
        (*self).try_into().ok()
    }
    fn to_i16(&self) -> Option<i16> {
        (*self).try_into().ok()
    }
    fn to_i32(&self) -> Option<i32> {
        (*self).try_into().ok()
    }
    fn to_i64(&self) -> Option<i64> {
        (*self).try_into().ok()
    }
    fn to_i128(&self) -> Option<i128> {
        Some(*self)
    }
    fn to_isize(&self) -> Option<isize> {
        (*self).try_into().ok()
    }

    fn to_u8(&self) -> Option<u8> {
        (*self).try_into().ok()
    }
    fn to_u16(&self) -> Option<u16> {
        (*self).try_into().ok()
    }
    fn to_u32(&self) -> Option<u32> {
        (*self).try_into().ok()
    }
    fn to_u64(&self) -> Option<u64> {
        (*self).try_into().ok()
    }
    fn to_u128(&self) -> Option<u128> {
        (*self).try_into().ok()
    }
    fn to_usize(&self) -> Option<usize> {
        (*self).try_into().ok()
    }

    fn to_f32(&self) -> Option<f32> {
        Some(*self as f32)
    }
    fn to_f64(&self) -> Option<f64> {
        Some(*self as f64)
    }
}

#[doc(hidden)]
/// Maintains the parsing state and context necessary to drive deserialization.
///
/// This struct tracks what the parser expects next, manages input position,
/// and remembers the span of the last processed token to provide accurate error reporting.
pub struct StackRunner<'input> {
    /// A version of the input that doesn't advance as we parse.
    pub original_input: &'input [u8],

    /// The raw input data being deserialized.
    pub input: &'input [u8],

    /// Stack of parsing instructions guiding the control flow.
    pub stack: Vec<Instruction>,

    /// Span of the last processed token, for accurate error reporting.
    pub last_span: Span,

    /// Format source identifier for error reporting
    pub format_source: &'static str,

    /// Array index tracking - maps depth to current index for arrays
    pub array_indices: Vec<usize>,

    /// Tuple variant field tracking - number of fields in current enum tuple variant
    pub enum_tuple_field_count: Option<usize>,

    /// Tuple variant field tracking - current field index being processed
    pub enum_tuple_current_field: Option<usize>,
}

impl<'input> StackRunner<'input> {
    /// Convenience function to create a DeserError using the original input and last_span.
    fn err(&self, kind: DeserErrorKind) -> DeserError<'input> {
        DeserError::new(kind, self.original_input, self.last_span)
    }

    /// Convenience function to create a DeserError from a ReflectError,
    /// using the original input and last_span for context.
    fn reflect_err(&self, err: ReflectError) -> DeserError<'input> {
        DeserError::new_reflect(err, self.original_input, self.last_span)
    }

    pub fn pop<'facet>(
        &mut self,
        mut wip: Partial<'facet>,
        reason: PopReason,
    ) -> Result<Partial<'facet>, DeserError<'input>> {
        trace!(
            "--- STACK has {:?} {}",
            self.stack.green(),
            "(POP)".bright_yellow()
        );
        trace!("Popping because {:?}", reason.yellow());

        let container_shape = wip.shape();
        match container_shape.ty {
            Type::User(UserType::Struct(sd)) => {
                let mut has_unset = false;

                trace!("Let's check all fields are initialized");
                for (index, field) in sd.fields.iter().enumerate() {
                    let is_set = wip.is_field_set(index).map_err(|err| {
                        trace!("Error checking field set status: {err:?}");
                        self.reflect_err(err)
                    })?;
                    if !is_set {
                        if field.flags.contains(FieldFlags::DEFAULT) {
                            wip.begin_nth_field(index)
                                .map_err(|e| self.reflect_err(e))?;

                            // Check for field-level default function first, then type-level default
                            if let Some(field_default_fn) = field.vtable.default_fn {
                                wip.set_field_default(field_default_fn)
                                    .map_err(|e| self.reflect_err(e))?;
                                trace!(
                                    "Field #{} {} @ {} was set to default value (via field default function)",
                                    index.yellow(),
                                    field.name.green(),
                                    field.offset.blue(),
                                );
                            } else if field.shape().is(Characteristic::Default) {
                                wip.set_default().map_err(|e| self.reflect_err(e))?;
                                trace!(
                                    "Field #{} {} @ {} was set to default value (via type default impl)",
                                    index.yellow(),
                                    field.name.green(),
                                    field.offset.blue(),
                                );
                            } else {
                                return Err(self.reflect_err(
                                    ReflectError::DefaultAttrButNoDefaultImpl {
                                        shape: field.shape(),
                                    },
                                ));
                            }
                            wip.end().map_err(|e| self.reflect_err(e))?;
                        } else {
                            trace!(
                                "Field #{} {} @ {} is not initialized",
                                index.yellow(),
                                field.name.green(),
                                field.offset.blue(),
                            );
                            has_unset = true;
                        }
                    }
                }

                if has_unset {
                    if container_shape.has_default_attr() {
                        // let's allocate and build a default value
                        let default_val = Partial::alloc_shape(container_shape)
                            .map_err(|e| self.reflect_err(e))?
                            .set_default()
                            .map_err(|e| self.reflect_err(e))?
                            .build()
                            .map_err(|e| self.reflect_err(e))?;
                        let peek = default_val.peek().into_struct().unwrap();

                        for (index, field) in sd.fields.iter().enumerate() {
                            let is_set = wip.is_field_set(index).map_err(|err| {
                                trace!("Error checking field set status: {err:?}");
                                self.reflect_err(err)
                            })?;
                            if !is_set {
                                trace!(
                                    "Field #{} {} @ {} is being set to default value (from default instance)",
                                    index.yellow(),
                                    field.name.green(),
                                    field.offset.blue(),
                                );
                                wip.begin_nth_field(index)
                                    .map_err(|e| self.reflect_err(e))?;
                                // Get the field as a Peek from the default value
                                let def_field = peek.field(index).unwrap();
                                wip.set_from_peek(&def_field)
                                    .map_err(|e| self.reflect_err(e))?;
                                wip.end().map_err(|e| self.reflect_err(e))?;
                            }
                        }
                    } else {
                        // Find the first uninitialized field to report in the error
                        for (index, field) in sd.fields.iter().enumerate() {
                            let is_set = wip.is_field_set(index).map_err(|err| {
                                trace!("Error checking field set status: {err:?}");
                                self.reflect_err(err)
                            })?;
                            if !is_set {
                                return Err(self.reflect_err(ReflectError::UninitializedField {
                                    shape: container_shape,
                                    field_name: field.name,
                                }));
                            }
                        }
                    }
                }
            }
            Type::User(UserType::Enum(ed)) => {
                trace!("Checking if enum is initialized correctly");

                // Check if a variant has been selected
                if let Some(variant) = wip.selected_variant() {
                    trace!("Variant {} is selected", variant.name.blue());

                    // Check if all fields in the variant are initialized
                    if !variant.data.fields.is_empty() {
                        let mut has_unset = false;

                        for (index, field) in variant.data.fields.iter().enumerate() {
                            let is_set = wip.is_field_set(index).map_err(|err| {
                                trace!("Error checking field set status: {err:?}");
                                self.reflect_err(err)
                            })?;

                            if !is_set {
                                if field.flags.contains(FieldFlags::DEFAULT) {
                                    wip.begin_nth_field(index)
                                        .map_err(|e| self.reflect_err(e))?;

                                    // Check for field-level default function first, then type-level default
                                    if let Some(field_default_fn) = field.vtable.default_fn {
                                        wip.set_field_default(field_default_fn)
                                            .map_err(|e| self.reflect_err(e))?;
                                        trace!(
                                            "Field #{} @ {} in variant {} was set to default value (via field default function)",
                                            index.yellow(),
                                            field.offset.blue(),
                                            variant.name
                                        );
                                    } else if field.shape().is(Characteristic::Default) {
                                        wip.set_default().map_err(|e| self.reflect_err(e))?;
                                        trace!(
                                            "Field #{} @ {} in variant {} was set to default value (via type default impl)",
                                            index.yellow(),
                                            field.offset.blue(),
                                            variant.name
                                        );
                                    } else {
                                        return Err(self.reflect_err(
                                            ReflectError::DefaultAttrButNoDefaultImpl {
                                                shape: field.shape(),
                                            },
                                        ));
                                    }
                                    wip.end().map_err(|e| self.reflect_err(e))?;
                                } else {
                                    trace!(
                                        "Field #{} @ {} in variant {} is not initialized",
                                        index.yellow(),
                                        field.offset.blue(),
                                        variant.name
                                    );
                                    has_unset = true;
                                }
                            }
                        }

                        if has_unset {
                            if container_shape.has_default_attr() {
                                trace!(
                                    "Enum has DEFAULT attr but variant has uninitialized fields"
                                );
                                // Handle similar to struct, allocate and build default value for variant
                                let default_val = Partial::alloc_shape(container_shape)
                                    .map_err(|e| self.reflect_err(e))?
                                    .set_default()
                                    .map_err(|e| self.reflect_err(e))?
                                    .build()
                                    .map_err(|e| self.reflect_err(e))?;

                                let peek = default_val.peek();
                                let peek_enum =
                                    peek.into_enum().map_err(|e| self.reflect_err(e))?;
                                let default_variant = peek_enum
                                    .active_variant()
                                    .map_err(|e| self.err(DeserErrorKind::VariantError(e)))?;

                                if default_variant.name == variant.name {
                                    // It's the same variant, fill in the missing fields
                                    for (index, _field) in variant.data.fields.iter().enumerate() {
                                        let is_set = wip.is_field_set(index).map_err(|err| {
                                            trace!("Error checking field set status: {err:?}");
                                            self.reflect_err(err)
                                        })?;
                                        if !is_set {
                                            if let Ok(Some(def_field)) = peek_enum.field(index) {
                                                wip.begin_nth_field(index)
                                                    .map_err(|e| self.reflect_err(e))?;
                                                wip.set_from_peek(&def_field)
                                                    .map_err(|e| self.reflect_err(e))?;
                                                wip.end().map_err(|e| self.reflect_err(e))?;
                                            }
                                        }
                                    }
                                }
                            } else {
                                // Find the first uninitialized field to report in the error
                                for (index, field) in variant.data.fields.iter().enumerate() {
                                    let is_set = wip.is_field_set(index).map_err(|err| {
                                        trace!("Error checking field set status: {err:?}");
                                        self.reflect_err(err)
                                    })?;
                                    if !is_set {
                                        return Err(self.reflect_err(
                                            ReflectError::UninitializedEnumField {
                                                shape: container_shape,
                                                variant_name: variant.name,
                                                field_name: field.name,
                                            },
                                        ));
                                    }
                                }
                            }
                        }
                    }
                } else if container_shape.has_default_attr() {
                    // No variant selected, but enum has default attribute - set to default
                    trace!("No variant selected but enum has DEFAULT attr; setting to default");
                    let default_val = Partial::alloc_shape(container_shape)
                        .map_err(|e| self.reflect_err(e))?
                        .set_default()
                        .map_err(|e| self.reflect_err(e))?
                        .build()
                        .map_err(|e| self.reflect_err(e))?;

                    let peek = default_val.peek();
                    let peek_enum = peek.into_enum().map_err(|e| self.reflect_err(e))?;
                    let default_variant_idx = peek_enum
                        .variant_index()
                        .map_err(|e| self.err(DeserErrorKind::VariantError(e)))?;

                    // Select the default variant
                    wip.select_nth_variant(default_variant_idx)
                        .map_err(|e| self.reflect_err(e))?;

                    // Copy all fields from default value
                    let variant = &ed.variants[default_variant_idx];
                    for (index, _field) in variant.data.fields.iter().enumerate() {
                        if let Ok(Some(def_field)) = peek_enum.field(index) {
                            wip.begin_nth_field(index)
                                .map_err(|e| self.reflect_err(e))?;
                            wip.set_from_peek(&def_field)
                                .map_err(|e| self.reflect_err(e))?;
                            wip.end().map_err(|e| self.reflect_err(e))?;
                        }
                    }
                }
            }
            _ => {
                trace!(
                    "Thing being popped is not a container I guess (it's a {}, innermost is {})",
                    wip.shape(),
                    wip.innermost_shape()
                );
            }
        }
        Ok(wip)
    }

    /// Internal common handler for GotScalar outcome, to deduplicate code.
    /// Helper to set numeric values with type conversion
    fn set_numeric_value<'facet>(
        &self,
        wip: &mut Partial<'facet>,
        value: &dyn NumericConvert,
    ) -> Result<(), DeserError<'input>>
    where
        'input: 'facet,
    {
        let shape = wip.innermost_shape();

        let Type::Primitive(PrimitiveType::Numeric(numeric_type)) = shape.ty else {
            return Err(self.err(DeserErrorKind::UnsupportedType {
                got: shape,
                wanted: "numeric type",
            }));
        };

        // Get the size from the layout
        let size_bytes = shape
            .layout
            .sized_layout()
            .map_err(|_| {
                self.err(DeserErrorKind::UnsupportedType {
                    got: shape,
                    wanted: "sized numeric type",
                })
            })?
            .size();

        if matches!(shape.def, Def::Scalar) {
            // Helper closure to convert and set numeric value
            macro_rules! convert_and_set {
                ($converter:expr, $target_type:expr) => {{
                    let converted = $converter.ok_or_else(|| {
                        self.err(DeserErrorKind::NumericConversion {
                            from: "numeric",
                            to: $target_type,
                        })
                    })?;
                    wip.set(converted).map_err(|e| self.reflect_err(e))?;
                }};
            }

            match numeric_type {
                NumericType::Integer { signed } => {
                    // First check if the shape is specifically usize or isize
                    if !signed && shape.is_type::<usize>() {
                        convert_and_set!(value.to_usize(), "usize")
                    } else if signed && shape.is_type::<isize>() {
                        convert_and_set!(value.to_isize(), "isize")
                    } else {
                        // Then check by size
                        match (size_bytes, signed) {
                            (1, true) => convert_and_set!(value.to_i8(), "i8"),
                            (2, true) => convert_and_set!(value.to_i16(), "i16"),
                            (4, true) => convert_and_set!(value.to_i32(), "i32"),
                            (8, true) => convert_and_set!(value.to_i64(), "i64"),
                            (16, true) => convert_and_set!(value.to_i128(), "i128"),
                            (1, false) => convert_and_set!(value.to_u8(), "u8"),
                            (2, false) => convert_and_set!(value.to_u16(), "u16"),
                            (4, false) => convert_and_set!(value.to_u32(), "u32"),
                            (8, false) => convert_and_set!(value.to_u64(), "u64"),
                            (16, false) => convert_and_set!(value.to_u128(), "u128"),
                            _ => {
                                return Err(self.err(DeserErrorKind::NumericConversion {
                                    from: "numeric",
                                    to: if signed {
                                        "unknown signed integer size"
                                    } else {
                                        "unknown unsigned integer size"
                                    },
                                }));
                            }
                        }
                    }
                }
                NumericType::Float => match size_bytes {
                    4 => convert_and_set!(value.to_f32(), "f32"),
                    8 => convert_and_set!(value.to_f64(), "f64"),
                    _ => {
                        return Err(self.err(DeserErrorKind::NumericConversion {
                            from: "numeric",
                            to: "unknown float size",
                        }));
                    }
                },
            }
        } else {
            // Not a scalar def - cannot convert
            return Err(self.err(DeserErrorKind::UnsupportedType {
                got: shape,
                wanted: "scalar type",
            }));
        }

        Ok(())
    }

    fn handle_scalar<'facet>(
        &self,
        wip: &mut Partial<'facet>,
        scalar: Scalar<'input>,
    ) -> Result<(), DeserError<'input>>
    where
        'input: 'facet, // 'input outlives 'facet
    {
        match scalar {
            Scalar::String(cow) => {
                match wip.innermost_shape().ty {
                    Type::User(UserType::Enum(_)) => {
                        if wip.selected_variant().is_some() {
                            // If we already have a variant selected, just put the string
                            wip.set(cow).map_err(|e| self.reflect_err(e))?;
                        } else {
                            // Try to select the variant
                            match wip.find_variant(&cow) {
                                Some((variant_index, _)) => {
                                    wip.select_nth_variant(variant_index)
                                        .map_err(|e| self.reflect_err(e))?;
                                }
                                None => {
                                    return Err(self.err(DeserErrorKind::NoSuchVariant {
                                        name: cow.to_string(),
                                        enum_shape: wip.innermost_shape(),
                                    }));
                                }
                            }
                        }
                    }
                    Type::Pointer(PointerType::Reference(_))
                        if wip.innermost_shape().is_type::<&str>() =>
                    {
                        // This is for handling the &str type
                        // The Cow may be Borrowed (we may have an owned string but need a &str)
                        match cow {
                            Cow::Borrowed(s) => wip.set(s).map_err(|e| self.reflect_err(e))?,
                            Cow::Owned(s) => wip.set(s).map_err(|e| self.reflect_err(e))?,
                        }; // Add semicolon to ignore the return value
                    }
                    _ => {
                        // Check if this is a scalar type that can be parsed from a string
                        let shape = wip.innermost_shape();
                        if matches!(shape.def, Def::Scalar) {
                            // Try parse_from_str for scalar types that might parse from strings
                            // (like IpAddr, UUID, Path, etc.)
                            match wip.parse_from_str(cow.as_ref()) {
                                Ok(_) => {
                                    // Successfully parsed
                                }
                                Err(parse_err) => {
                                    // Parsing failed - check if it's because parse isn't supported
                                    // or if parsing actually failed
                                    match parse_err {
                                        ReflectError::OperationFailed {
                                            shape: _,
                                            operation,
                                        } if operation.contains("does not support parsing") => {
                                            // Type doesn't have a parse function, try direct conversion
                                            wip.set(cow.to_string())
                                                .map_err(|e| self.reflect_err(e))?;
                                        }
                                        _ => {
                                            // Actual parsing failure
                                            return Err(self.err(DeserErrorKind::ReflectError(
                                                ReflectError::OperationFailed {
                                                    shape,
                                                    operation: "Failed to parse string value",
                                                },
                                            )));
                                        }
                                    }
                                }
                            }
                        } else {
                            // Not a scalar, just set as String
                            wip.set(cow.to_string()).map_err(|e| self.reflect_err(e))?;
                        }
                    }
                }
            }
            Scalar::U64(value) => {
                self.set_numeric_value(wip, &value)?;
            }
            Scalar::I64(value) => {
                self.set_numeric_value(wip, &value)?;
            }
            Scalar::F64(value) => {
                self.set_numeric_value(wip, &value)?;
            }
            Scalar::U128(value) => {
                self.set_numeric_value(wip, &value)?;
            }
            Scalar::I128(value) => {
                self.set_numeric_value(wip, &value)?;
            }
            Scalar::Bool(value) => {
                wip.set(value).map_err(|e| self.reflect_err(e))?;
            }
            Scalar::Null => {
                wip.set_default().map_err(|e| self.reflect_err(e))?;
            }
        }
        Ok(())
    }

    /// Handle value parsing
    fn value<'facet>(
        &mut self,
        mut wip: Partial<'facet>,
        outcome: Spanned<Outcome<'input>>,
    ) -> Result<Partial<'facet>, DeserError<'input>>
    where
        'input: 'facet, // 'input must outlive 'facet
    {
        trace!(
            "--- STACK has {:?} {}",
            self.stack.green(),
            "(VALUE)".bright_yellow()
        );

        let original_shape = wip.shape();
        trace!("Handling value of type {}", original_shape.blue());

        // Handle null values
        if matches!(outcome.node, Outcome::Scalar(Scalar::Null)) {
            wip.set_default().map_err(|e| self.reflect_err(e))?;
            return Ok(wip);
        }

        // Resolve the innermost value to deserialize
        let mut smart_pointer_begun = false;
        loop {
            trace!("  Loop iteration: current shape is {}", wip.shape().blue());
            if matches!(wip.shape().def, Def::Option(_)) {
                trace!("  Starting Some(_) option for {}", wip.shape().blue());
                wip.begin_some().map_err(|e| self.reflect_err(e))?;
                self.stack.push(Instruction::Pop(PopReason::Some));
            } else if let Def::Pointer(inner) = wip.shape().def {
                // Check if we've already begun this smart pointer
                // (this can happen with slice pointees where the shape doesn't change)
                if smart_pointer_begun {
                    break;
                }
                if let Some(pointee) = inner.pointee() {
                    trace!(
                        "  Starting smart pointer for {} (pointee is {})",
                        wip.shape().blue(),
                        pointee.yellow(),
                    );
                } else {
                    trace!(
                        "  Starting smart pointer for {} (no pointee)",
                        wip.shape().blue()
                    );
                }
                trace!("  About to call begin_smart_ptr()");
                wip.begin_smart_ptr().map_err(|e| self.reflect_err(e))?;
                trace!(
                    "  After begin_smart_ptr(), shape is now {}",
                    wip.shape().blue()
                );
                self.stack.push(Instruction::Pop(PopReason::SmartPointer));
                smart_pointer_begun = true;
            } else if let Some(inner_fn) = wip.shape().inner {
                let inner = inner_fn();
                trace!(
                    "  Starting wrapped value for {} (inner is {})",
                    wip.shape().blue(),
                    inner.yellow()
                );
                wip.begin_inner().map_err(|e| self.reflect_err(e))?;
                self.stack.push(Instruction::Pop(PopReason::Wrapper));
            } else {
                break;
            }
        }

        if wip.shape() != original_shape {
            trace!(
                "Handling shape {} as innermost {}",
                original_shape.blue(),
                wip.shape().yellow()
            );
        }

        match outcome.node {
            Outcome::Scalar(s) => {
                trace!("Parsed scalar value: {}", s.cyan());
                self.handle_scalar(&mut wip, s)?;
            }
            Outcome::ListStarted => {
                let shape = wip.innermost_shape();

                // First check if this is a tuple struct (including empty tuples)
                if let Type::User(UserType::Struct(st)) = shape.ty {
                    if st.kind == StructKind::Tuple {
                        trace!(
                            "Array starting for tuple struct ({}) with {} fields!",
                            shape.blue(),
                            st.fields.len()
                        );

                        // Non-empty tuples need to process list events
                        trace!("Beginning pushback");
                        self.stack.push(Instruction::ListItemOrListClose);
                        return Ok(wip);
                    }
                }

                match shape.def {
                    Def::Array(_) => {
                        trace!("Array starting for array ({})!", shape.blue());
                        // We'll initialize the array elements one by one through the pushback workflow
                        // Don't call put_default, as arrays need different initialization
                    }
                    Def::Slice(_) => {
                        trace!("Array starting for slice ({})!", shape.blue());
                    }
                    Def::List(_) => {
                        trace!("Array starting for list ({})!", shape.blue());
                        wip.set_default().map_err(|e| self.reflect_err(e))?;
                    }
                    _ => {
                        // Check if we're building a smart pointer slice
                        if matches!(shape.def, Def::Pointer(_)) && smart_pointer_begun {
                            trace!("Array starting for smart pointer slice ({})!", shape.blue());
                            wip.begin_list().map_err(|e| self.reflect_err(e))?;
                        } else if let Type::User(user_ty) = shape.ty {
                            // For non-collection types, check the Type enum
                            match user_ty {
                                UserType::Enum(_) => {
                                    trace!("Array starting for enum ({})!", shape.blue());
                                    // Check if we have a tuple variant selected
                                    if let Some(variant) = wip.selected_variant() {
                                        use facet_core::StructKind;
                                        if variant.data.kind == StructKind::Tuple {
                                            // For tuple variants, we'll handle array elements as tuple fields
                                            // Initialize tuple field tracking
                                            self.enum_tuple_field_count =
                                                Some(variant.data.fields.len());
                                            self.enum_tuple_current_field = Some(0);
                                        } else {
                                            return Err(self.err(DeserErrorKind::UnsupportedType {
                                                got: shape,
                                                wanted: "tuple variant for array deserialization",
                                            }));
                                        }
                                    } else {
                                        return Err(self.err(DeserErrorKind::UnsupportedType {
                                            got: shape,
                                            wanted: "enum with variant selected",
                                        }));
                                    }
                                }
                                UserType::Struct(_) => {
                                    // Regular struct shouldn't be parsed from array
                                    // (Tuples are already handled above)
                                    return Err(self.err(DeserErrorKind::UnsupportedType {
                                        got: shape,
                                        wanted: "array, list, tuple, or slice",
                                    }));
                                }
                                _ => {
                                    return Err(self.err(DeserErrorKind::UnsupportedType {
                                        got: shape,
                                        wanted: "array, list, tuple, or slice",
                                    }));
                                }
                            }
                        } else {
                            return Err(self.err(DeserErrorKind::UnsupportedType {
                                got: shape,
                                wanted: "array, list, tuple, or slice",
                            }));
                        }
                    }
                }
                trace!("Beginning pushback");
                self.stack.push(Instruction::ListItemOrListClose);

                // Only call begin_list() for actual lists, not arrays
                match shape.def {
                    Def::List(_) => {
                        wip.begin_list().map_err(|e| self.reflect_err(e))?;
                    }
                    Def::Array(_) => {
                        // Arrays don't need begin_list()
                        // Initialize index tracking for this array
                        self.array_indices.push(0);
                    }
                    Def::Slice(_) => {
                        // Slices don't need begin_list()
                        // They will be populated element by element
                    }
                    _ => {
                        // For other types like tuples, no special initialization needed
                    }
                }
            }
            Outcome::ListEnded => {
                trace!("List closing");
                // Clean up array index tracking if this was an array
                let shape = wip.shape();
                if matches!(shape.def, Def::Array(_)) {
                    self.array_indices.pop();
                }
                wip.end().map_err(|e| self.reflect_err(e))?;
            }
            Outcome::ObjectStarted => {
                let shape = wip.shape();
                match shape.def {
                    Def::Map(_md) => {
                        trace!("Object starting for map value ({})!", shape.blue());
                        wip.begin_map().map_err(|e| self.reflect_err(e))?;
                    }
                    _ => {
                        // For non-collection types, check the Type enum
                        if let Type::User(user_ty) = shape.ty {
                            match user_ty {
                                UserType::Enum(_) => {
                                    trace!("Object starting for enum value ({})!", shape.blue());
                                    // nothing to do here
                                }
                                UserType::Struct(_) => {
                                    trace!("Object starting for struct value ({})!", shape.blue());
                                    // nothing to do here
                                }
                                _ => {
                                    return Err(self.err(DeserErrorKind::UnsupportedType {
                                        got: shape,
                                        wanted: "map, enum, or struct",
                                    }));
                                }
                            }
                        } else if let Type::User(UserType::Struct(struct_type)) = shape.ty {
                            if struct_type.kind == StructKind::Tuple {
                                // This could be a tuple that was serialized as an object
                                // Despite this being unusual, we'll handle it here for robustness
                                trace!(
                                    "Object starting for tuple ({}) with {} fields - unusual but handling",
                                    shape.blue(),
                                    struct_type.fields.len()
                                );
                                // Tuples are treated as structs
                            }
                        } else {
                            return Err(self.err(DeserErrorKind::UnsupportedType {
                                got: shape,
                                wanted: "map, enum, struct, or tuple",
                            }));
                        }
                    }
                }

                self.stack.push(Instruction::ObjectKeyOrObjectClose);
            }
            Outcome::ObjectEnded => todo!(),
        }
        Ok(wip)
    }

    fn object_key_or_object_close<'facet>(
        &mut self,
        mut wip: Partial<'facet>,
        outcome: Spanned<Outcome<'input>>,
    ) -> Result<Partial<'facet>, DeserError<'input>>
    where
        'input: 'facet,
    {
        trace!(
            "STACK: {:?} {}",
            self.stack.green(),
            "(OK/OC)".bright_yellow()
        );
        match outcome.node {
            Outcome::Scalar(Scalar::String(key)) => {
                trace!("Parsed object key: {}", key.cyan());

                let mut ignore = false;
                let mut needs_pop = true;
                let mut handled_by_flatten = false;

                let shape = wip.innermost_shape();
                match shape.ty {
                    Type::User(UserType::Struct(sd)) => {
                        // First try to find a direct field match
                        if let Some(index) = wip.field_index(&key) {
                            trace!("It's a struct field");
                            wip.begin_nth_field(index)
                                .map_err(|e| self.reflect_err(e))?;
                        } else {
                            trace!(
                                "Did not find direct field match in innermost shape {}",
                                shape.blue()
                            );

                            // Check for flattened fields
                            let mut found_in_flatten = false;
                            for (index, field) in sd.fields.iter().enumerate() {
                                if field.flags.contains(FieldFlags::FLATTEN) {
                                    trace!("Found flattened field #{index}");
                                    // Enter the flattened field
                                    wip.begin_nth_field(index)
                                        .map_err(|e| self.reflect_err(e))?;

                                    // Check if this flattened field has the requested key
                                    if let Some(subfield_index) = wip.field_index(&key) {
                                        trace!("Found key {key} in flattened field");
                                        wip.begin_nth_field(subfield_index)
                                            .map_err(|e| self.reflect_err(e))?;
                                        found_in_flatten = true;
                                        handled_by_flatten = true;
                                        break;
                                    } else if let Some((_variant_index, _variant)) =
                                        wip.find_variant(&key)
                                    {
                                        trace!("Found key {key} in flattened field");
                                        wip.select_variant_named(&key)
                                            .map_err(|e| self.reflect_err(e))?;
                                        found_in_flatten = true;
                                        break;
                                    } else {
                                        // Key not in this flattened field, go back up
                                        wip.end().map_err(|e| self.reflect_err(e))?;
                                    }
                                }
                            }

                            if !found_in_flatten {
                                if wip.shape().has_deny_unknown_fields_attr() {
                                    trace!(
                                        "It's not a struct field AND we're denying unknown fields"
                                    );
                                    return Err(self.err(DeserErrorKind::UnknownField {
                                        field_name: key.to_string(),
                                        shape: wip.shape(),
                                    }));
                                } else {
                                    trace!(
                                        "It's not a struct field and we're ignoring unknown fields"
                                    );
                                    ignore = true;
                                }
                            }
                        }
                    }
                    Type::User(UserType::Enum(_ed)) => match wip.find_variant(&key) {
                        Some((index, variant)) => {
                            trace!(
                                "Selecting variant {}::{}",
                                wip.shape().blue(),
                                variant.name.yellow(),
                            );
                            wip.select_nth_variant(index)
                                .map_err(|e| self.reflect_err(e))?;

                            // Let's see what's in the variant — if it's tuple-like with only one field, we want to push field 0
                            if matches!(variant.data.kind, StructKind::Tuple)
                                && variant.data.fields.len() == 1
                            {
                                trace!(
                                    "Tuple variant {}::{} encountered, pushing field 0",
                                    wip.shape().blue(),
                                    variant.name.yellow()
                                );
                                wip.begin_nth_field(0).map_err(|e| self.reflect_err(e))?;
                                self.stack.push(Instruction::Pop(PopReason::ObjectVal));
                            }

                            needs_pop = false;
                        }
                        None => {
                            if let Some(_variant_index) = wip.selected_variant() {
                                trace!(
                                    "Already have a variant selected, treating {} as struct field of {}::{}",
                                    key,
                                    wip.shape().blue(),
                                    wip.selected_variant().unwrap().name.yellow(),
                                );
                                // Try to find the field index of the key within the selected variant
                                if let Some(index) = wip.field_index(&key) {
                                    trace!("Found field {} in selected variant", key.blue());
                                    wip.begin_nth_field(index)
                                        .map_err(|e| self.reflect_err(e))?;
                                } else if wip.shape().has_deny_unknown_fields_attr() {
                                    trace!("Unknown field in variant and denying unknown fields");
                                    return Err(self.err(DeserErrorKind::UnknownField {
                                        field_name: key.to_string(),
                                        shape: wip.shape(),
                                    }));
                                } else {
                                    trace!(
                                        "Ignoring unknown field '{}' in variant '{}::{}'",
                                        key,
                                        wip.shape(),
                                        wip.selected_variant().unwrap().name
                                    );
                                    ignore = true;
                                }
                            } else {
                                return Err(self.err(DeserErrorKind::NoSuchVariant {
                                    name: key.to_string(),
                                    enum_shape: wip.shape(),
                                }));
                            }
                        }
                    },
                    _ => {
                        // Check if it's a map
                        if let Def::Map(map_def) = shape.def {
                            wip.begin_key().map_err(|e| self.reflect_err(e))?;

                            // Check if the map key type is transparent (has an inner shape)
                            let key_shape = map_def.k();
                            if key_shape.inner.is_some() {
                                // For transparent types, we need to navigate into the inner type
                                // The inner type should be String for JSON object keys
                                // Use begin_inner for consistency with begin_* naming convention
                                wip.begin_inner().map_err(|e| self.reflect_err(e))?;
                                wip.set(key.to_string()).map_err(|e| self.reflect_err(e))?;
                                wip.end().map_err(|e| self.reflect_err(e))?; // End inner
                            } else {
                                // For non-transparent types, set the string directly
                                wip.set(key.to_string()).map_err(|e| self.reflect_err(e))?;
                            }

                            wip.end().map_err(|e| self.reflect_err(e))?; // Complete the key frame
                            wip.begin_value().map_err(|e| self.reflect_err(e))?;
                        } else {
                            return Err(self.err(DeserErrorKind::Unimplemented(
                                "object key for non-struct/map",
                            )));
                        }
                    }
                }

                self.stack.push(Instruction::ObjectKeyOrObjectClose);
                if ignore {
                    self.stack.push(Instruction::SkipValue);
                } else {
                    if needs_pop && !handled_by_flatten {
                        trace!("Pushing Pop insn to stack (ObjectVal)");
                        self.stack.push(Instruction::Pop(PopReason::ObjectVal));
                    } else if handled_by_flatten {
                        // For flattened fields, we only need one pop for the field itself.
                        // The flattened struct should remain active until the outer object is finished.
                        trace!("Pushing Pop insn to stack (ObjectVal) for flattened field");
                        self.stack.push(Instruction::Pop(PopReason::ObjectVal));
                    }
                    self.stack.push(Instruction::Value(ValueReason::ObjectVal));
                }
                Ok(wip)
            }
            Outcome::ObjectEnded => {
                trace!("Object closing");
                Ok(wip)
            }
            _ => Err(self.err(DeserErrorKind::UnexpectedOutcome {
                got: outcome.node.into_owned(),
                wanted: "scalar or object close",
            })),
        }
    }

    fn list_item_or_list_close<'facet>(
        &mut self,
        mut wip: Partial<'facet>,
        outcome: Spanned<Outcome<'input>>,
    ) -> Result<Partial<'facet>, DeserError<'input>>
    where
        'input: 'facet,
    {
        trace!(
            "--- STACK has {:?} {}",
            self.stack.green(),
            "(LI/LC)".bright_yellow()
        );
        match outcome.node {
            Outcome::ListEnded => {
                trace!("List close");
                // Clean up array index tracking if this was an array
                let shape = wip.shape();
                if matches!(shape.def, Def::Array(_)) {
                    self.array_indices.pop();
                }

                // Clean up enum tuple variant tracking if this was an enum tuple
                if let Type::User(UserType::Enum(_)) = shape.ty {
                    if self.enum_tuple_field_count.is_some() {
                        trace!("Enum tuple variant list ended");
                        self.enum_tuple_field_count = None;
                        self.enum_tuple_current_field = None;
                    }
                }

                // Special case: if we're at an empty tuple, we've successfully parsed it
                if let Type::User(UserType::Struct(st)) = shape.ty {
                    if st.kind == StructKind::Tuple && st.fields.is_empty() {
                        trace!("Empty tuple parsed from []");
                        // The empty tuple is complete - no fields to initialize
                    }
                }

                // Don't end the list here - let the Pop instruction handle it
                Ok(wip)
            }
            _ => {
                self.stack.push(Instruction::ListItemOrListClose);
                self.stack.push(Instruction::Pop(PopReason::ListVal));

                trace!(
                    "Expecting list item, doing a little push before doing value with outcome {}",
                    outcome.magenta()
                );
                trace!("Before push, wip.shape is {}", wip.shape().blue());

                // Different handling for arrays vs lists
                let shape = wip.shape();
                match shape.def {
                    Def::Array(ad) => {
                        // Arrays use the last index in our tracking vector
                        if let Some(current_index) = self.array_indices.last().copied() {
                            // Check bounds
                            if current_index >= ad.n {
                                return Err(self.err(DeserErrorKind::ArrayOverflow {
                                    shape,
                                    max_len: ad.n,
                                }));
                            }

                            // Set this array element
                            wip.begin_nth_element(current_index)
                                .map_err(|e| self.reflect_err(e))?;

                            // Increment the index for next time
                            if let Some(last) = self.array_indices.last_mut() {
                                *last += 1;
                            }
                        } else {
                            // This shouldn't happen if we properly initialize in ListStarted
                            return Err(self.err(DeserErrorKind::Unimplemented(
                                "Array index tracking not initialized",
                            )));
                        }
                    }
                    Def::List(_) => {
                        wip.begin_list_item().map_err(|e| self.reflect_err(e))?;
                    }
                    _ => {
                        // Check if this is a smart pointer with slice pointee
                        if matches!(shape.def, Def::Pointer(_)) {
                            trace!("List item for smart pointer slice");
                            wip.begin_list_item().map_err(|e| self.reflect_err(e))?;
                        }
                        // Check if this is an enum tuple variant
                        else if let Type::User(UserType::Enum(_)) = shape.ty {
                            if let (Some(field_count), Some(current_field)) =
                                (self.enum_tuple_field_count, self.enum_tuple_current_field)
                            {
                                if current_field >= field_count {
                                    // Too many elements for this tuple variant
                                    return Err(self.err(DeserErrorKind::ArrayOverflow {
                                        shape,
                                        max_len: field_count,
                                    }));
                                }

                                // Process this tuple field
                                wip.begin_nth_enum_field(current_field)
                                    .map_err(|e| self.reflect_err(e))?;

                                // Advance to next field
                                self.enum_tuple_current_field = Some(current_field + 1);
                            } else {
                                return Err(self.err(DeserErrorKind::UnsupportedType {
                                    got: shape,
                                    wanted: "enum with tuple variant selected",
                                }));
                            }
                        }
                        // Check if this is a tuple
                        else if let Type::User(UserType::Struct(struct_type)) = shape.ty {
                            if struct_type.kind == StructKind::Tuple {
                                // Tuples use field indexing
                                // Find the next uninitialized field
                                let mut field_index = None;
                                for i in 0..struct_type.fields.len() {
                                    if !wip.is_field_set(i).map_err(|e| self.reflect_err(e))? {
                                        field_index = Some(i);
                                        break;
                                    }
                                }

                                if let Some(idx) = field_index {
                                    wip.begin_nth_field(idx).map_err(|e| self.reflect_err(e))?;
                                } else {
                                    // All fields are set, this is too many elements
                                    return Err(self.err(DeserErrorKind::ArrayOverflow {
                                        shape,
                                        max_len: struct_type.fields.len(),
                                    }));
                                }
                            } else {
                                // Not a tuple struct
                                return Err(self.err(DeserErrorKind::UnsupportedType {
                                    got: shape,
                                    wanted: "array, list, or tuple",
                                }));
                            }
                        } else {
                            // Not a struct type at all
                            return Err(self.err(DeserErrorKind::UnsupportedType {
                                got: shape,
                                wanted: "array, list, or tuple",
                            }));
                        }
                    }
                }

                trace!(" After push, wip.shape is {}", wip.shape().cyan());

                // Special handling: if we're now at an empty tuple and we see a list start,
                // we can handle the flexible coercion from []
                if matches!(outcome.node, Outcome::ListStarted) {
                    if let Type::User(UserType::Struct(st)) = wip.shape().ty {
                        if st.kind == StructKind::Tuple && st.fields.is_empty() {
                            trace!(
                                "Empty tuple field with list start - initializing empty tuple and expecting immediate close"
                            );
                            // Initialize the empty tuple with default value since it has no fields to fill
                            wip.set_default().map_err(|e| self.reflect_err(e))?;
                            // Continue processing - we still need to handle the list close
                        }
                    }
                }

                wip = self.value(wip, outcome)?;
                Ok(wip)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
