// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! Enough YAML to read a file this crate also writes the exporter for.
//!
//! Block mappings, block sequences, flow sequences and scalars, indented with spaces. That is
//! what `tools/package_common.py` emits and what a [`Manifest`](crate::Manifest) is; everything
//! else in the language -- anchors, flow mappings, tags, multiple documents, block scalars -- is
//! refused rather than half-supported. The one thing worse than a format this cannot read is one
//! it reads differently from everybody else.
//!
//! Every scalar stays the text it was written as. `steps: 8` is the string `"8"` here and becomes
//! a number where it is used, because what a value means is the thing reading it's to say: the
//! same `320,640,1280` is three numbers to the U-Net and a string to anything else.

use std::collections::BTreeMap;

use crate::error::{Error, Result};

/// One value, at whatever depth.
#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    Scalar(String),
    Map(BTreeMap<String, Node>),
    Seq(Vec<Node>),
}

impl Node {
    pub fn into_map(self, what: &str) -> Result<BTreeMap<String, Node>> {
        match self {
            Node::Map(map) => Ok(map),
            _ => Err(Error::format(format!("{what} is not a block of keys"))),
        }
    }

    pub fn into_seq(self, what: &str) -> Result<Vec<Node>> {
        match self {
            Node::Seq(items) => Ok(items),
            _ => Err(Error::format(format!("{what} is not a list"))),
        }
    }

    pub fn into_scalar(self, what: &str) -> Result<String> {
        match self {
            Node::Scalar(text) => Ok(text),
            _ => Err(Error::format(format!("{what} is not a single value"))),
        }
    }

    /// This value as text, if it is one.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Node::Scalar(text) => Some(text),
            _ => None,
        }
    }

    /// The items of this value, if it is a list.
    pub fn as_seq(&self) -> Option<&[Node]> {
        match self {
            Node::Seq(items) => Some(items),
            _ => None,
        }
    }
}

/// One line that is neither blank nor a comment: how far it is indented, what is on it, and which
/// line of the file it was, for the complaints.
struct Row<'a> {
    indent: usize,
    text: &'a str,
    number: usize,
}

/// The whole document, which is one block at indent zero.
pub fn parse(text: &str) -> Result<Node> {
    let mut rows = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let number = index + 1;
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        // A tab is not indentation in YAML, and a file that used one would nest differently here
        // than it does anywhere else. Better to say so than to guess a width for it.
        let indent = line.len() - line.trim_start_matches(' ').len();
        if line[indent..].starts_with('\t') {
            return Err(Error::format(format!(
                "line {number}: indented with a tab, and indentation here is spaces"
            )));
        }
        rows.push(Row {
            indent,
            text: line[indent..].trim_end(),
            number,
        });
    }

    if rows.is_empty() {
        return Err(Error::format("there is nothing in this file"));
    }
    if rows[0].indent != 0 {
        return Err(Error::format(format!(
            "line {}: the file starts indented",
            rows[0].number
        )));
    }

    let mut at = 0;
    let node = block(&rows, &mut at, 0)?;
    if at != rows.len() {
        return Err(Error::format(format!(
            "line {}: this is indented less than the block it is in",
            rows[at].number
        )));
    }
    Ok(node)
}

/// Every row at `indent`, as one mapping or one sequence.
///
/// Which of the two it is, the first row says: `- ` begins a sequence and anything else a
/// mapping. Rows indented further belong to the value above them and are read by the call this
/// one makes; a row indented less ends the block and is the caller's.
fn block(rows: &[Row], at: &mut usize, indent: usize) -> Result<Node> {
    match is_item(rows[*at].text) {
        true => sequence(rows, at, indent),
        false => mapping(rows, at, indent),
    }
}

/// Whether a row is a `- ` list item rather than a key.
fn is_item(text: &str) -> bool {
    text == "-" || text.starts_with("- ")
}

fn mapping(rows: &[Row], at: &mut usize, indent: usize) -> Result<Node> {
    let mut map: BTreeMap<String, Node> = BTreeMap::new();

    while *at < rows.len() && rows[*at].indent >= indent {
        let row = &rows[*at];
        if row.indent > indent {
            return Err(Error::format(format!(
                "line {}: this is indented further than the key above it",
                row.number
            )));
        }
        if is_item(row.text) {
            return Err(Error::format(format!(
                "line {}: a list item where a key was expected",
                row.number
            )));
        }

        let (key, rest) = split_key(row.text, row.number)?;
        let number = row.number;
        *at += 1;

        let value = if rest.is_empty() {
            // `key:` with nothing after it: what it holds is the block below, which has to be
            // indented further than the key that introduced it.
            if *at < rows.len() && rows[*at].indent > indent {
                let deeper = rows[*at].indent;
                block(rows, at, deeper)?
            } else {
                return Err(Error::format(format!(
                    "line {number}: {key:?} is given nothing"
                )));
            }
        } else {
            value(rest, number)?
        };

        if map.insert(key.clone(), value).is_some() {
            return Err(Error::format(format!(
                "line {number}: {key:?} is written twice"
            )));
        }
    }

    Ok(Node::Map(map))
}

fn sequence(rows: &[Row], at: &mut usize, indent: usize) -> Result<Node> {
    let mut items = Vec::new();

    while *at < rows.len() && rows[*at].indent >= indent {
        let row = &rows[*at];
        if row.indent > indent {
            return Err(Error::format(format!(
                "line {}: this is indented further than the item above it",
                row.number
            )));
        }
        if row.text == "-" {
            return Err(Error::format(format!(
                "line {}: a list item with nothing in it",
                row.number
            )));
        }
        let Some(rest) = row.text.strip_prefix("- ") else {
            return Err(Error::format(format!(
                "line {}: a key where a list item was expected",
                row.number
            )));
        };

        let number = row.number;
        *at += 1;
        items.push(value(rest, number)?);
    }

    Ok(Node::Seq(items))
}

/// A `key:` and whatever followed it on the line.
///
/// The colon is the first one outside quotes and outside brackets, so a quoted key holding one is
/// still a key. It has to be followed by a space or by nothing: `a:b` is the scalar `a:b` in
/// YAML, and reading it as a key here would be reading this file differently from every other
/// reader.
fn split_key(text: &str, number: usize) -> Result<(String, &str)> {
    let mut quote: Option<u8> = None;
    let mut depth = 0usize;

    for (index, byte) in text.bytes().enumerate() {
        match quote {
            Some(open) if byte == open => quote = None,
            Some(_) => {}
            None if byte == b'"' || byte == b'\'' => quote = Some(byte),
            None if byte == b'[' => depth += 1,
            None if byte == b']' => depth = depth.saturating_sub(1),
            None if byte == b':' && depth == 0 => {
                let after = &text[index + 1..];
                if !after.is_empty() && !after.starts_with(' ') {
                    continue;
                }
                let key = plain_or_quoted(text[..index].trim_end(), number)?;
                if key.is_empty() {
                    return Err(Error::format(format!("line {number}: a key with no name")));
                }
                return Ok((key, after.trim_start()));
            }
            None => {}
        }
    }
    Err(Error::format(format!(
        "line {number}: {text:?} is neither a key nor a list item"
    )))
}

/// One value as written after a `key:` or a `- `: a flow sequence, or a scalar.
fn value(text: &str, number: usize) -> Result<Node> {
    let text = text.trim();
    if text.starts_with('[') {
        let (node, rest) = flow(text, number)?;
        // Trailing text after a closed bracket is a line that means something else somewhere
        // else, and guessing which would be the wrong kind of helpful.
        let rest = strip_comment(rest);
        if !rest.trim().is_empty() {
            return Err(Error::format(format!(
                "line {number}: {rest:?} follows a list and is not part of it"
            )));
        }
        return Ok(node);
    }
    Ok(Node::Scalar(plain_or_quoted(text, number)?))
}

/// A `[a, b, [c, d]]`, from its opening bracket, and whatever is left of the line after it.
fn flow(text: &str, number: usize) -> Result<(Node, &str)> {
    let mut rest = text
        .strip_prefix('[')
        .expect("a flow sequence starts with its bracket")
        .trim_start();
    let mut items = Vec::new();

    if let Some(after) = rest.strip_prefix(']') {
        return Ok((Node::Seq(items), after));
    }

    loop {
        let (item, after) = if rest.starts_with('[') {
            flow(rest, number)?
        } else {
            let end = flow_item_ends(rest, number)?;
            (
                Node::Scalar(plain_or_quoted(rest[..end].trim(), number)?),
                &rest[end..],
            )
        };
        items.push(item);

        rest = after.trim_start();
        if let Some(after) = rest.strip_prefix(',') {
            rest = after.trim_start();
            // `[1, 2, ]` -- a comma before the close is a trailing comma, not an empty item.
            if let Some(after) = rest.strip_prefix(']') {
                return Ok((Node::Seq(items), after));
            }
            continue;
        }
        if let Some(after) = rest.strip_prefix(']') {
            return Ok((Node::Seq(items), after));
        }
        return Err(Error::format(format!(
            "line {number}: a list that is not closed"
        )));
    }
}

/// Where one scalar item of a flow sequence ends: at the comma or the bracket that follows it,
/// whichever comes first outside quotes.
fn flow_item_ends(text: &str, number: usize) -> Result<usize> {
    let mut quote: Option<u8> = None;
    for (index, byte) in text.bytes().enumerate() {
        match quote {
            Some(open) if byte == open => quote = None,
            Some(_) => {}
            None if byte == b'"' || byte == b'\'' => quote = Some(byte),
            None if byte == b',' || byte == b']' => return Ok(index),
            None => {}
        }
    }
    Err(Error::format(format!(
        "line {number}: a list that is not closed"
    )))
}

/// A scalar with its quotes taken off, or a plain one with its comment taken off.
fn plain_or_quoted(text: &str, number: usize) -> Result<String> {
    let text = text.trim();

    if let Some(body) = quoted(text, b'\'') {
        // A doubled quote is the only escape a single quoted scalar has, so a lone one is a value
        // that ended earlier than it looks like it did.
        if body.replace("''", "").contains('\'') {
            return Err(Error::format(format!(
                "line {number}: a quote inside a single quoted value has to be doubled"
            )));
        }
        return Ok(body.replace("''", "'"));
    }

    if let Some(body) = quoted(text, b'"') {
        return escapes(body, number);
    }

    if text.starts_with('"') || text.starts_with('\'') {
        return Err(Error::format(format!(
            "line {number}: {text:?} opens a quote it does not close"
        )));
    }

    Ok(strip_comment(text).trim_end().to_string())
}

/// What is left of a plain scalar once a trailing comment is taken off.
///
/// A `#` only begins a comment when a space separates it from what came before, which is the rule
/// everywhere else: `tag#1` is a value holding a hash, and `tag #1` is the value `tag` with a
/// comment after it. A value that really does want a space and then a hash has to be quoted, and
/// the exporter quotes it -- see `_quote` in `tools/package_common.py`.
fn strip_comment(text: &str) -> &str {
    for (index, _) in text.match_indices('#') {
        if index == 0 || text.as_bytes()[index - 1] == b' ' {
            return &text[..index];
        }
    }
    text
}

/// The inside of `text`, if the whole of it is wrapped in `mark`.
fn quoted(text: &str, mark: u8) -> Option<&str> {
    let bytes = text.as_bytes();
    match bytes.len() >= 2 && bytes[0] == mark && bytes[bytes.len() - 1] == mark {
        true => Some(&text[1..text.len() - 1]),
        false => None,
    }
}

/// What a double quoted scalar means, which is the only place a backslash does anything.
fn escapes(body: &str, number: usize) -> Result<String> {
    let mut text = String::new();
    let mut chars = body.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            text.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => text.push('\n'),
            Some('t') => text.push('\t'),
            Some('r') => text.push('\r'),
            Some('"') => text.push('"'),
            Some('\'') => text.push('\''),
            Some('\\') => text.push('\\'),
            Some('/') => text.push('/'),
            Some(other) => {
                return Err(Error::format(format!(
                    "line {number}: \\{other} is not an escape this reads"
                )))
            }
            None => {
                return Err(Error::format(format!(
                    "line {number}: a value ends in a backslash"
                )))
            }
        }
    }
    Ok(text)
}
