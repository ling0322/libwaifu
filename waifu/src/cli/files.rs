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

//! Finding a file on the disk, as a box that opens over whatever screen asked for it.
//!
//! Not a screen of its own, unlike the model picker next door: that one owns the terminal and runs
//! its own loop, because it happens once before there is anything else to be looking at. This one
//! is a piece a screen already up puts on top of itself, so it owns nothing -- the host keeps it
//! in an `Option`, hands it the keys while it is there, gives it an area to draw in, and takes the
//! path back out of [`Outcome`].
//!
//! That is the whole of the interface, and it is why this is worth having as a module rather than
//! as part of the drawing screen: the next thing that wants a file off the disk needs three lines
//! rather than another list.

use std::cell::Cell;
use std::fs;
use std::path::{Path, PathBuf};

use ratatui::crossterm::event::{KeyCode, KeyEvent};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Clear, Paragraph};
use ratatui::Frame;

/// The name a parent directory goes by, and the row that walks to it.
const PARENT: &str = "..";

/// How far page up and page down move.
///
/// A page really means the height of the box, which only the drawing knows; a key press does not.
/// Ten is a page on most terminals and a reasonable jump on the rest, which is closer than moving
/// by one and the only thing that can be said without asking the screen.
const PAGE: usize = 10;

/// The widest the box gets, before the room it is given decides.
const MAX_WIDTH: u16 = 78;

/// The fewest and the most rows of list it is drawn with.
///
/// Between the two it is as tall as what is in the directory, so that a directory of three things
/// is a box of three rather than a box of twenty with a corner used. The count is of everything in
/// the directory rather than of what the filter leaves, so that typing narrows the list without
/// the box moving under it.
const MIN_ROWS: u16 = 3;
const MAX_ROWS: u16 = 18;

/// What the border, the path and the foot cost on top of the rows.
const CHROME: u16 = 4;

/// One row of the list.
#[derive(Clone, Debug, Eq, PartialEq)]
struct Entry {
    name: String,
    directory: bool,
}

/// What a key press left the picker in.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Outcome {
    /// Still open. The host keeps drawing it and keeps handing it keys.
    Open,
    /// Closed with nothing chosen.
    Cancelled,
    /// Closed on this file. Absolute, so that the host need not know where it was opened.
    Picked(PathBuf),
}

/// A list of what is in one directory, with somewhere to walk from it.
pub struct FilePicker {
    /// What the box is called, which is what the host is asking for rather than where it is.
    title: String,
    /// The directory being looked at. Absolute, so the title says something unambiguous and the
    /// path handed back does not depend on where the process happens to be standing.
    directory: PathBuf,
    /// Everything in it worth offering: `..` first where there is a parent, then directories,
    /// then files, each run sorted by name without regard to case.
    entries: Vec<Entry>,
    /// Where the keys are, counted over the rows actually shown rather than over `entries`.
    selected: usize,
    /// What has been typed to narrow the list.
    filter: String,
    /// The suffixes a file has to end in to be offered, lowercased. Empty offers every file.
    suffixes: Vec<String>,
    /// Why the directory could not be read, where it could not.
    trouble: Option<String>,
    /// The row the list is scrolled to.
    ///
    /// In a Cell because it is a property of the drawing rather than of the choice: what it has to
    /// be depends on how tall the box was drawn, which only the drawing knows, and a key press
    /// that moves the cursor has no way to work it out.
    first: Cell<usize>,
}

impl FilePicker {
    /// A picker over the directory `start` is in, or over `start` itself where it is one.
    ///
    /// `suffixes` are the file endings to offer, as `["png", "jpg"]`; an empty list offers every
    /// file. Directories are always offered, whatever they are called, since walking through them
    /// is how anything else is reached.
    pub fn open(title: &str, start: &Path, suffixes: &[&str]) -> FilePicker {
        let mut picker = FilePicker {
            title: title.to_string(),
            directory: PathBuf::new(),
            entries: Vec::new(),
            selected: 0,
            filter: String::new(),
            suffixes: suffixes.iter().map(|s| s.to_lowercase()).collect(),
            trouble: None,
            first: Cell::new(0),
        };

        // A file rather than a directory means someone typed a path in the box before opening
        // this, and what they want to see is where that file lives with that file marked.
        let (directory, mark) = match start.is_dir() {
            true => (start.to_path_buf(), None),
            false => (
                start.parent().unwrap_or(Path::new(".")).to_path_buf(),
                start
                    .file_name()
                    .map(|name| name.to_string_lossy().to_string()),
            ),
        };

        picker.walk_to(&directory);
        if let Some(mark) = mark {
            picker.select(&mark);
        }

        picker
    }

    /// Where the picker is looking, for a host that wants to say so.
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// Reads `directory` and starts the list at the top of it.
    ///
    /// A directory that cannot be read leaves the list empty rather than refusing to move: what
    /// is on screen then is the name of the directory and why it could not be opened, with `..`
    /// still there to walk back out by.
    fn walk_to(&mut self, directory: &Path) {
        // Made absolute without asking the filesystem, so that a path through a symbolic link
        // still reads as the one that was walked rather than as the one it landed on.
        self.directory = std::path::absolute(directory).unwrap_or_else(|_| directory.to_path_buf());
        self.filter.clear();
        self.selected = 0;
        self.first.set(0);
        self.trouble = None;

        let mut directories = Vec::new();
        let mut files = Vec::new();
        match fs::read_dir(&self.directory) {
            Ok(reading) => {
                for entry in reading.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();

                    // Dot files are not offered. Nobody is looking for one here, and a home
                    // directory is mostly them.
                    if name.starts_with('.') {
                        continue;
                    }

                    // A symbolic link is followed to decide which half it belongs in, and one
                    // that points nowhere is neither, so it is left out.
                    match entry.path().is_dir() {
                        true => directories.push(Entry {
                            name,
                            directory: true,
                        }),
                        false if self.offers(&name) => files.push(Entry {
                            name,
                            directory: false,
                        }),
                        false => {}
                    }
                }
            }
            Err(error) => self.trouble = Some(error.to_string()),
        }

        let by_name = |a: &Entry, b: &Entry| a.name.to_lowercase().cmp(&b.name.to_lowercase());
        directories.sort_by(by_name);
        files.sort_by(by_name);

        self.entries.clear();
        if self.directory.parent().is_some() {
            self.entries.push(Entry {
                name: PARENT.to_string(),
                directory: true,
            });
        }
        self.entries.append(&mut directories);
        self.entries.append(&mut files);
    }

    /// Whether a file of this name is one of the kinds asked for.
    fn offers(&self, name: &str) -> bool {
        if self.suffixes.is_empty() {
            return true;
        }

        let name = name.to_lowercase();
        self.suffixes
            .iter()
            .any(|suffix| name.ends_with(&format!(".{suffix}")))
    }

    /// Puts the cursor on the row called `name`, where there is one.
    fn select(&mut self, name: &str) {
        if let Some(index) = self.shown().iter().position(|entry| entry.name == name) {
            self.selected = index;
        }
    }

    /// The rows the filter leaves, which is what the cursor counts over and what is drawn.
    fn shown(&self) -> Vec<&Entry> {
        if self.filter.is_empty() {
            return self.entries.iter().collect();
        }

        // Anywhere in the name rather than at the start: what is being typed is usually the
        // middle of a file name that begins with a date or a camera's prefix.
        let wanted = self.filter.to_lowercase();
        self.entries
            .iter()
            .filter(|entry| entry.name.to_lowercase().contains(&wanted))
            .collect()
    }

    /// Walks into the row the cursor is on, or picks it where it is a file.
    fn enter(&mut self) -> Outcome {
        let shown = self.shown();
        let Some(entry) = shown.get(self.selected) else {
            return Outcome::Open;
        };

        if !entry.directory {
            return Outcome::Picked(self.directory.join(&entry.name));
        }

        if entry.name == PARENT {
            self.leave();
        } else {
            let into = self.directory.join(&entry.name);
            self.walk_to(&into);
        }

        Outcome::Open
    }

    /// Walks out to the parent, with the cursor left on the directory just left.
    fn leave(&mut self) {
        let Some(parent) = self.directory.parent().map(Path::to_path_buf) else {
            return;
        };

        let left = self
            .directory
            .file_name()
            .map(|name| name.to_string_lossy().to_string());

        self.walk_to(&parent);
        if let Some(left) = left {
            self.select(&left);
        }
    }

    fn move_by(&mut self, rows: isize) {
        let last = self.shown().len().saturating_sub(1);
        self.selected = (self.selected as isize + rows).clamp(0, last as isize) as usize;
    }

    /// What one key press does.
    ///
    /// The host hands over every key while the picker is open and reads the answer: [`Outcome::Open`]
    /// means keep it there, and either of the other two means take it down.
    pub fn key(&mut self, key: KeyEvent) -> Outcome {
        match key.code {
            KeyCode::Esc => return Outcome::Cancelled,
            KeyCode::Enter => return self.enter(),

            KeyCode::Up => self.move_by(-1),
            KeyCode::Down => self.move_by(1),
            KeyCode::PageUp => self.move_by(-(PAGE as isize)),
            KeyCode::PageDown => self.move_by(PAGE as isize),
            KeyCode::Home => self.selected = 0,
            KeyCode::End => self.selected = self.shown().len().saturating_sub(1),

            // The two directions of walking, on the two keys that already mean them. Right only
            // descends: a file is chosen with enter, so that one key does not both open a
            // directory and end the whole picker depending on what the cursor happens to be on.
            KeyCode::Left => self.leave(),
            KeyCode::Right => {
                if self.shown().get(self.selected).is_some_and(|e| e.directory) {
                    return self.enter();
                }
            }

            // Backspace undoes what was typed, and once there is nothing left to undo it walks
            // out of the directory -- which is the order a shell does the same two things in.
            KeyCode::Backspace => {
                if self.filter.pop().is_none() {
                    self.leave();
                }
                self.selected = 0;
            }
            KeyCode::Char(character) => {
                self.filter.push(character);
                self.selected = 0;
            }
            _ => {}
        }

        Outcome::Open
    }

    /// Draws the picker in the middle of `area`, over whatever was there.
    ///
    /// `area` is the room it may use rather than the box itself: it takes the smaller of that and
    /// a size that reads well, so a host can hand it the whole frame without the list becoming a
    /// wall on a wide terminal.
    pub fn render(&self, frame: &mut Frame, area: Rect) {
        let rows = (self.entries.len() as u16).clamp(MIN_ROWS, MAX_ROWS);
        let box_area = centred(area, MAX_WIDTH, rows + CHROME);
        frame.render_widget(Clear, box_area);

        let outline = Block::bordered()
            .border_type(BorderType::Thick)
            .border_style(Style::new().fg(Color::Yellow))
            .title(Span::styled(
                format!(" {} ", self.title),
                Style::new().fg(Color::Yellow),
            ));
        let inner = outline.inner(box_area);
        frame.render_widget(outline, box_area);
        if inner.height < 3 {
            return;
        }

        let [where_at, rows, foot] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(1),
        ])
        .areas(inner);

        // The tail of the path rather than the head: which directory this is is the part that
        // changes as the keys walk, and the part a long path pushes off the screen.
        frame.render_widget(
            Paragraph::new(Line::from(
                Span::raw(tail(
                    &self.directory.to_string_lossy(),
                    where_at.width as usize,
                ))
                .bold(),
            )),
            where_at,
        );

        frame.render_widget(self.list(rows.height as usize), rows);
        frame.render_widget(self.foot(), foot);
    }

    /// The rows, scrolled only as far as it takes to keep the cursor on screen.
    fn list(&self, height: usize) -> Paragraph<'static> {
        let shown = self.shown();
        if let Some(trouble) = &self.trouble {
            return Paragraph::new(Line::from(
                Span::raw(format!("cannot be read: {trouble}")).fg(Color::Red),
            ));
        }
        if shown.is_empty() {
            return Paragraph::new(Line::from(Span::raw("nothing here to draw from").dim()));
        }

        let height = height.max(1);
        let mut first = self.first.get().min(shown.len().saturating_sub(1));
        first = first.min(self.selected);
        if self.selected >= first + height {
            first = self.selected + 1 - height;
        }
        self.first.set(first);

        let lines: Vec<Line> = shown
            .iter()
            .enumerate()
            .skip(first)
            .take(height)
            .map(|(index, entry)| {
                let on = index == self.selected;
                let name = match entry.directory {
                    true => format!("{}/", entry.name),
                    false => entry.name.clone(),
                };

                Line::from(vec![
                    Span::raw(if on { "> " } else { "  " }),
                    Span::styled(name, row_style(on, entry.directory)),
                ])
            })
            .collect();

        Paragraph::new(lines)
    }

    /// What was typed, and what the keys do -- the second only while there is room for it.
    fn foot(&self) -> Paragraph<'static> {
        if !self.filter.is_empty() {
            let shown = self.shown().len();
            return Paragraph::new(Line::from(vec![
                Span::raw("/").fg(Color::Yellow),
                Span::raw(self.filter.clone()).fg(Color::Yellow),
                Span::raw(format!("   {shown} of {}", self.entries.len())).dim(),
            ]));
        }

        Paragraph::new(
            Line::from(vec![
                Span::raw("enter").bold(),
                Span::raw(" open or pick  "),
                Span::raw("←→").bold(),
                Span::raw(" in and out  "),
                Span::raw("type").bold(),
                Span::raw(" to narrow  "),
                Span::raw("esc").bold(),
                Span::raw(" close"),
            ])
            .style(Style::new().fg(Color::DarkGray)),
        )
    }
}

/// How a row reads: marked where the cursor is on it, and directories apart from files, since
/// which of the two a row is decides what enter does to it.
fn row_style(on: bool, directory: bool) -> Style {
    match (on, directory) {
        (true, _) => Style::default().fg(Color::Black).bg(Color::Cyan),
        (false, true) => Style::default().fg(Color::Cyan),
        (false, false) => Style::default(),
    }
}

/// A rectangle of at most `width` by `height`, in the middle of `area`.
fn centred(area: Rect, width: u16, height: u16) -> Rect {
    let width = width.min(area.width);
    let height = height.min(area.height);

    Rect {
        x: area.x + (area.width - width) / 2,
        y: area.y + (area.height - height) / 2,
        width,
        height,
    }
}

/// The last `width` characters of `text`, marked as cut where anything was.
fn tail(text: &str, width: usize) -> String {
    let characters: Vec<char> = text.chars().collect();
    if characters.len() <= width {
        return text.to_string();
    }
    if width <= 1 {
        return "…".chars().take(width).collect();
    }

    std::iter::once('…')
        .chain(characters[characters.len() - (width - 1)..].iter().copied())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use ratatui::backend::TestBackend;
    use ratatui::crossterm::event::{KeyEventKind, KeyModifiers};

    /// A directory tree to walk, under a name of its own so that two tests do not share one.
    struct Tree(PathBuf);

    impl Tree {
        fn new(name: &str) -> Tree {
            let root = std::env::temp_dir().join(format!("waifu-files-{name}"));
            let _ = fs::remove_dir_all(&root);
            fs::create_dir_all(root.join("pictures")).unwrap();
            fs::create_dir_all(root.join("models")).unwrap();

            for name in ["cat.png", "Dog.JPG", "notes.txt", "cat.png.bak"] {
                fs::write(root.join(name), b"x").unwrap();
            }
            fs::write(root.join("pictures").join("inside.png"), b"x").unwrap();
            fs::write(root.join(".hidden.png"), b"x").unwrap();

            Tree(root)
        }

        fn picker(&self) -> FilePicker {
            FilePicker::open("pick one", &self.0, &["png", "jpg"])
        }
    }

    impl Drop for Tree {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn type_in(picker: &mut FilePicker, text: &str) {
        for character in text.chars() {
            assert_eq!(picker.key(press(KeyCode::Char(character))), Outcome::Open);
        }
    }

    fn names(picker: &FilePicker) -> Vec<String> {
        picker.shown().iter().map(|e| e.name.clone()).collect()
    }

    /// The whole box, drawn into a buffer, as one long string of what it says.
    fn screen(picker: &FilePicker, width: u16, height: u16) -> String {
        let mut terminal = ratatui::Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal
            .draw(|frame| picker.render(frame, frame.area()))
            .unwrap();

        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect()
    }

    #[test]
    fn directories_come_first_and_only_the_kinds_asked_for_follow() {
        let tree = Tree::new("sorted");
        let picker = tree.picker();

        // The parent, then the directories, then the files -- and of the four files only the two
        // that are pictures. "cat.png.bak" ends in .bak whatever is in the middle of it.
        assert_eq!(
            names(&picker),
            vec!["..", "models", "pictures", "cat.png", "Dog.JPG"]
        );
    }

    #[test]
    fn a_suffix_is_matched_whatever_case_it_is_written_in() {
        // "Dog.JPG" above is offered by a picker asking for "jpg", which is the case a camera
        // and a phone between them will produce.
        let tree = Tree::new("case");
        assert!(names(&tree.picker()).contains(&"Dog.JPG".to_string()));
    }

    #[test]
    fn no_suffixes_offers_every_file() {
        let tree = Tree::new("everything");
        let picker = FilePicker::open("pick one", &tree.0, &[]);

        assert!(names(&picker).contains(&"notes.txt".to_string()));
        // Dot files stay out either way. Nobody is looking for one here.
        assert!(!names(&picker).contains(&".hidden.png".to_string()));
    }

    #[test]
    fn enter_walks_into_a_directory_and_picks_a_file() {
        let tree = Tree::new("walking");
        let mut picker = tree.picker();

        // Onto "pictures", which is the third row.
        picker.key(press(KeyCode::Down));
        picker.key(press(KeyCode::Down));
        assert_eq!(picker.key(press(KeyCode::Enter)), Outcome::Open);
        assert_eq!(picker.directory(), tree.0.join("pictures"));
        assert_eq!(names(&picker), vec!["..", "inside.png"]);

        picker.key(press(KeyCode::Down));
        assert_eq!(
            picker.key(press(KeyCode::Enter)),
            Outcome::Picked(tree.0.join("pictures").join("inside.png"))
        );
    }

    #[test]
    fn walking_back_out_leaves_the_cursor_on_where_it_came_from() {
        // The thing that makes a list worth walking rather than typing: coming out of a directory
        // puts the cursor back on it, so the one below it is one key away.
        let tree = Tree::new("back-out");
        let mut picker = tree.picker();

        picker.key(press(KeyCode::Down));
        picker.key(press(KeyCode::Down));
        picker.key(press(KeyCode::Enter));
        picker.key(press(KeyCode::Left));

        assert_eq!(picker.directory(), tree.0);
        assert_eq!(names(&picker)[picker.selected], "pictures");
    }

    #[test]
    fn typing_narrows_the_list_and_backspace_widens_it_again() {
        let tree = Tree::new("narrow");
        let mut picker = tree.picker();

        // Anywhere in the name, not just at the start.
        type_in(&mut picker, "og");
        assert_eq!(names(&picker), vec!["Dog.JPG"]);

        // And without regard to case, in both directions.
        picker.key(press(KeyCode::Backspace));
        picker.key(press(KeyCode::Backspace));
        type_in(&mut picker, "CAT");
        assert_eq!(names(&picker), vec!["cat.png"]);

        assert_eq!(
            picker.key(press(KeyCode::Enter)),
            Outcome::Picked(tree.0.join("cat.png"))
        );
    }

    #[test]
    fn backspace_with_nothing_typed_walks_out_the_way_a_shell_does() {
        let tree = Tree::new("backspace-out");
        let mut picker = tree.picker();

        type_in(&mut picker, "pic");
        picker.key(press(KeyCode::Enter));
        assert_eq!(picker.directory(), tree.0.join("pictures"));

        // Nothing typed in here, so the next one is the other thing backspace means.
        picker.key(press(KeyCode::Backspace));
        assert_eq!(picker.directory(), tree.0);
    }

    #[test]
    fn the_cursor_stops_at_both_ends() {
        let tree = Tree::new("ends");
        let mut picker = tree.picker();
        let last = names(&picker).len() - 1;

        for _ in 0..50 {
            picker.key(press(KeyCode::Down));
        }
        assert_eq!(picker.selected, last);

        for _ in 0..50 {
            picker.key(press(KeyCode::Up));
        }
        assert_eq!(picker.selected, 0);

        picker.key(press(KeyCode::End));
        assert_eq!(picker.selected, last);
        picker.key(press(KeyCode::Home));
        assert_eq!(picker.selected, 0);

        // A page past the end is the end, not a panic.
        picker.key(press(KeyCode::PageDown));
        assert_eq!(picker.selected, last);
        picker.key(press(KeyCode::PageUp));
        assert_eq!(picker.selected, 0);
    }

    #[test]
    fn opening_on_a_file_shows_the_directory_it_is_in_with_it_marked() {
        let tree = Tree::new("on-a-file");
        let picker = FilePicker::open("pick one", &tree.0.join("Dog.JPG"), &["png", "jpg"]);

        assert_eq!(picker.directory(), tree.0);
        assert_eq!(names(&picker)[picker.selected], "Dog.JPG");
    }

    #[test]
    fn esc_closes_it_without_picking_anything() {
        let tree = Tree::new("esc");
        let mut picker = tree.picker();
        assert_eq!(picker.key(press(KeyCode::Esc)), Outcome::Cancelled);
    }

    #[test]
    fn a_path_that_is_not_there_yet_opens_where_it_would_be() {
        // Someone typed a name into the box before making the file, or before the run that will
        // write it. The directory it would be in is the useful thing to show, and it is what
        // opening on a file that does exist shows as well.
        let tree = Tree::new("not-yet");
        let picker = FilePicker::open("pick one", &tree.0.join("later.png"), &["png"]);

        assert_eq!(picker.directory(), tree.0);
        assert!(names(&picker).contains(&"cat.png".to_string()));
    }

    #[test]
    fn a_directory_that_cannot_be_read_says_so_rather_than_showing_an_empty_list() {
        // A path that runs through a file rather than a directory, which is what a typo in the
        // middle of one looks like. read_dir refuses it, and what is on screen is why.
        let tree = Tree::new("unreadable");
        let picker = FilePicker::open("pick one", &tree.0.join("cat.png").join("nested"), &["png"]);

        let drawn = screen(&picker, 80, 24);
        assert!(drawn.contains("cannot be read"), "{drawn}");

        // And there is still a way out of it.
        assert_eq!(names(&picker), vec![".."]);
    }

    #[test]
    fn a_filter_that_matches_nothing_says_so_rather_than_looking_broken() {
        let tree = Tree::new("nothing");
        let mut picker = tree.picker();
        type_in(&mut picker, "zzzz");

        let drawn = screen(&picker, 80, 24);
        assert!(drawn.contains("nothing here"), "{drawn}");
        // And enter on no row at all does nothing rather than picking the directory.
        assert_eq!(picker.key(press(KeyCode::Enter)), Outcome::Open);
    }

    #[test]
    fn what_is_on_screen_is_where_it_is_and_what_is_in_it() {
        let tree = Tree::new("screen");
        let mut picker = tree.picker();
        let drawn = screen(&picker, 80, 24);

        for wanted in ["pick one", "pictures/", "cat.png", "enter", "esc"] {
            assert!(drawn.contains(wanted), "the box does not say {wanted:?}");
        }
        // Directories are marked as such, files are not.
        assert!(!drawn.contains("cat.png/"), "{drawn}");

        // What was typed replaces the key hints, with how much of the list is left.
        type_in(&mut picker, "cat");
        let drawn = screen(&picker, 80, 24);
        assert!(drawn.contains("/cat"), "{drawn}");
        assert!(drawn.contains("1 of 5"), "{drawn}");
    }

    #[test]
    fn the_list_scrolls_only_as_far_as_it_takes_to_keep_the_cursor_on_it() {
        let root = std::env::temp_dir().join("waifu-files-scrolling");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        for index in 0..40 {
            fs::write(root.join(format!("{index:03}.png")), b"x").unwrap();
        }

        let mut picker = FilePicker::open("pick one", &root, &["png"]);
        let short = |picker: &FilePicker| screen(picker, 40, 12);

        // The top of the list to start with, and the top row still on screen.
        assert!(short(&picker).contains("000.png"), "{}", short(&picker));

        for _ in 0..30 {
            picker.key(press(KeyCode::Down));
        }
        let drawn = short(&picker);
        assert!(!drawn.contains("000.png"), "{drawn}");
        assert!(drawn.contains("029.png"), "{drawn}");

        // Back to the top, and so is the window: it followed rather than staying where it was.
        picker.key(press(KeyCode::Home));
        assert!(short(&picker).contains("000.png"), "{}", short(&picker));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_box_with_no_room_in_it_still_draws() {
        // A terminal too small to put a list in is not a reason to panic on a subtraction.
        let tree = Tree::new("cramped");
        let picker = tree.picker();
        for (width, height) in [(1, 1), (4, 3), (12, 5), (0, 0)] {
            screen(&picker, width.max(1), height.max(1));
        }
    }

    #[test]
    fn a_long_path_is_cut_at_the_front_where_the_part_that_matters_is_the_end() {
        assert_eq!(tail("/home/someone/pictures", 40), "/home/someone/pictures");
        assert_eq!(tail("/home/someone/pictures", 10), "…/pictures");
        assert_eq!(tail("/home/someone/pictures", 1), "…");
        assert_eq!(tail("", 4), "");
    }

    #[test]
    fn a_key_it_has_no_use_for_leaves_it_where_it_was() {
        let tree = Tree::new("unknown-key");
        let mut picker = tree.picker();
        let before = names(&picker);

        assert_eq!(
            picker.key(KeyEvent::new(KeyCode::F(5), KeyModifiers::NONE)),
            Outcome::Open
        );
        assert_eq!(picker.selected, 0);
        assert_eq!(names(&picker), before);

        // And a key event carrying a kind other than a press is the host's to filter, not this
        // one's: what arrives here is acted on.
        let _ = KeyEventKind::Press;
    }
}
