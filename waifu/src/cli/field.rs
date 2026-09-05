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

//! A box you can type a prompt into.
//!
//! Small enough to write out rather than depend on an editor widget: a run of characters, a place
//! in it, and the arithmetic that says where the lines break so that the cursor can be put where
//! the character it is on was drawn.

use std::ops::Range;

use ratatui::crossterm::event::KeyCode;

/// Text being edited, held as characters rather than bytes so that a position is a position.
#[derive(Clone, Debug, Default)]
pub struct TextField {
    characters: Vec<char>,
    cursor: usize,
}

impl TextField {
    /// A field that already has something in it, with the cursor after it.
    ///
    /// A screen starts every box empty but one: the picture to draw from, where `-i` names a file
    /// before there is a screen to type it into.
    pub fn new(text: &str) -> TextField {
        let characters: Vec<char> = text.chars().collect();
        TextField {
            cursor: characters.len(),
            characters,
        }
    }

    pub fn text(&self) -> String {
        self.characters.iter().collect()
    }

    /// Whether the cursor is before the first character, with nothing to its left.
    pub fn at_start(&self) -> bool {
        self.cursor == 0
    }

    /// Whether the cursor is after the last one, with nothing to its right.
    pub fn at_end(&self) -> bool {
        self.cursor == self.characters.len()
    }

    pub fn insert(&mut self, character: char) {
        self.characters.insert(self.cursor, character);
        self.cursor += 1;
    }

    /// Deletes the character before the cursor.
    pub fn backspace(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            self.characters.remove(self.cursor);
        }
    }

    /// Deletes the character the cursor is on.
    pub fn delete(&mut self) {
        if self.cursor < self.characters.len() {
            self.characters.remove(self.cursor);
        }
    }

    pub fn left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    pub fn right(&mut self) {
        self.cursor = (self.cursor + 1).min(self.characters.len());
    }

    pub fn home(&mut self) {
        self.cursor = 0;
    }

    pub fn end(&mut self) {
        self.cursor = self.characters.len();
    }

    /// The text either side of the cursor, and the character it is sitting on if there is one.
    ///
    /// For a box that draws its own cursor rather than asking the terminal for one. Where the
    /// cursor is has to come out of the middle of the text: a mark stuck on the end says the
    /// cursor is somewhere it is not, from the first press of left or home onwards.
    pub fn around_the_cursor(&self) -> (String, Option<char>, String) {
        let before = self.characters[..self.cursor].iter().collect();
        let on = self.characters.get(self.cursor).copied();
        let after = match on {
            Some(_) => self.characters[self.cursor + 1..].iter().collect(),
            // Past the last character there is nothing to be on and nothing after it.
            None => String::new(),
        };
        (before, on, after)
    }

    /// Where the lines break when the text is drawn `width` columns wide.
    ///
    /// Every character lands on exactly one line, the spaces a line was broken at included, so
    /// that a position in the text is a position on a line rather than something that has to be
    /// searched for.
    fn breaks(&self, width: usize) -> Vec<Range<usize>> {
        let width = width.max(1);
        let mut lines = Vec::new();

        let mut start = 0;
        while start + width < self.characters.len() {
            let window = &self.characters[start..start + width];
            // Broken after the last space that fits, so that a word is not cut in half; a word
            // too long to break inside is cut, as there is nowhere else to put it.
            let end = match window.iter().rposition(|character| *character == ' ') {
                Some(space) => start + space + 1,
                None => start + width,
            };
            lines.push(start..end);
            start = end;
        }
        lines.push(start..self.characters.len());

        lines
    }

    /// The text as the lines it is drawn as, and the row and column the cursor is on.
    pub fn wrapped(&self, width: usize) -> (Vec<String>, (usize, usize)) {
        let width = width.max(1);
        let breaks = self.breaks(width);
        let mut lines: Vec<String> = breaks
            .iter()
            .map(|line| self.characters[line.clone()].iter().collect())
            .collect();

        let mut row = breaks
            .iter()
            .position(|line| self.cursor < line.end)
            .unwrap_or(breaks.len() - 1);
        let mut column = self.cursor - breaks[row].start;

        // At the end of a line that is exactly full there is no column left to sit in, so the
        // cursor waits at the start of the line the next character will begin.
        if column >= width {
            row += 1;
            column = 0;
            lines.push(String::new());
        }

        (lines, (row, column))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn types_and_deletes_where_the_cursor_is() {
        let mut field = TextField::new("cat");
        field.insert('s');
        assert_eq!(field.text(), "cats");

        field.left();
        field.left();
        field.insert('r');
        assert_eq!(field.text(), "carts");

        field.backspace();
        assert_eq!(field.text(), "cats");
        field.delete();
        assert_eq!(field.text(), "cas");
    }

    #[test]
    fn the_cursor_stops_at_both_ends() {
        let mut field = TextField::new("ab");
        field.home();
        field.left();
        field.backspace();
        assert_eq!(field.text(), "ab");

        field.end();
        field.right();
        field.delete();
        assert_eq!(field.text(), "ab");
    }

    #[test]
    fn counts_characters_rather_than_bytes() {
        let mut field = TextField::new("猫耳");
        field.backspace();
        assert_eq!(field.text(), "猫");
        field.insert('娘');
        assert_eq!(field.text(), "猫娘");
    }

    #[test]
    fn says_what_is_either_side_of_the_cursor() {
        let mut field = TextField::new("30");

        // Past the end there is no character to be on, which is what a box draws its mark for.
        assert_eq!(
            field.around_the_cursor(),
            ("30".to_string(), None, String::new())
        );

        field.left();
        assert_eq!(
            field.around_the_cursor(),
            ("3".to_string(), Some('0'), String::new())
        );

        field.home();
        assert_eq!(
            field.around_the_cursor(),
            (String::new(), Some('3'), "0".to_string())
        );

        // Every character is accounted for exactly once, wherever the cursor is, so that drawing
        // the three pieces draws the text.
        let (before, on, after) = field.around_the_cursor();
        assert_eq!(
            format!("{before}{}{after}", on.unwrap_or_default()),
            field.text()
        );
    }

    #[test]
    fn breaks_lines_at_spaces() {
        let field = TextField::new("a photo of an astronaut riding a horse");
        let (lines, _) = field.wrapped(12);

        assert_eq!(
            lines,
            ["a photo of ", "an ", "astronaut ", "riding a ", "horse"]
        );
        // Every character is still there, in order, which is what the cursor arithmetic rests on.
        assert_eq!(lines.concat(), field.text());
    }

    #[test]
    fn a_word_too_long_to_break_is_cut() {
        let field = TextField::new("aaaaaaaa");
        let (lines, _) = field.wrapped(3);
        assert_eq!(lines, ["aaa", "aaa", "aa"]);
    }

    #[test]
    fn says_which_row_and_column_the_cursor_is_on() {
        let mut field = TextField::new("hello world");
        assert_eq!(field.wrapped(6).1, (1, 5), "at the end of the second line");

        field.home();
        assert_eq!(field.wrapped(6).1, (0, 0));

        // The character after the space the line broke at is the first of the next line.
        for _ in 0..6 {
            field.right();
        }
        assert_eq!(field.wrapped(6).1, (1, 0));
    }

    #[test]
    fn a_cursor_at_the_end_of_a_full_line_waits_on_the_next_one() {
        let field = TextField::new("abcdef");
        let (lines, cursor) = field.wrapped(6);
        assert_eq!(lines, ["abcdef", ""]);
        assert_eq!(cursor, (1, 0));
    }

    #[test]
    fn an_empty_field_is_one_empty_line() {
        let field = TextField::default();
        assert_eq!(field.wrapped(10), (vec![String::new()], (0, 0)));

        // A box with no room in it still has to say where the cursor went.
        assert_eq!(TextField::new("ab").wrapped(0).1, (2, 0));
    }
}

/// The keys that edit a box of text, taking only the characters `accepts` allows.
///
/// Left and right move the cursor here rather than moving between boxes, which is the one place
/// the screens they are on give those two keys away: in a box you type into they cannot mean
/// anything else.
pub fn edit_text(field: &mut TextField, code: KeyCode, accepts: fn(&char) -> bool) {
    match code {
        KeyCode::Char(character) if accepts(&character) => field.insert(character),
        KeyCode::Backspace => field.backspace(),
        KeyCode::Delete => field.delete(),
        KeyCode::Left => field.left(),
        KeyCode::Right => field.right(),
        KeyCode::Home => field.home(),
        KeyCode::End => field.end(),
        _ => {}
    }
}
