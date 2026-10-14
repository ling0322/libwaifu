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

using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Windows.Media;
using System.Windows.Threading;
using Libwaifu.Gui.Engine;

namespace Libwaifu.Gui.ViewModels
{
    /// <summary>A recording that has been handed to the screen, and how long it runs.</summary>
    public class Clip : Observable
    {
        public Clip(string path)
        {
            Path = path;
            Name = System.IO.Path.GetFileName(path);

            // The engine takes samples and not a file, so a recording is read the moment it is
            // handed over rather than when a job is asked for: whether it can be read at all is
            // worth knowing before the button is pressed.
            string trouble;
            Sound = Engine.Sound.Read(path, out trouble);
            Trouble = trouble;
        }

        public string Path { get; private set; }

        public string Name { get; private set; }

        /// <summary>The samples, or null when it could not be read.</summary>
        public Recording Sound { get; private set; }

        public string Trouble { get; private set; }

        public bool Readable
        {
            get { return Sound != null; }
        }
    }

    /// <summary>
    /// The player under a clip: play, pause, a slider that can be dragged, and the two ends of
    /// the time. It never starts on its own, and stops at the end rather than looping.
    /// </summary>
    public class PlayerViewModel : Observable
    {
        private readonly MediaPlayer player = new MediaPlayer();
        private readonly DispatcherTimer tick;
        private bool playing;
        private bool dragging;
        private double length;
        private string trouble;

        public PlayerViewModel()
        {
            tick = new DispatcherTimer(DispatcherPriority.Background)
            {
                Interval = TimeSpan.FromMilliseconds(100),
            };
            tick.Tick += delegate
            {
                if (!dragging)
                {
                    Raise("Position");
                    Raise("PositionText");
                }
            };

            player.MediaOpened += delegate
            {
                length = player.NaturalDuration.HasTimeSpan
                    ? player.NaturalDuration.TimeSpan.TotalSeconds
                    : 0;
                Raise("Length");
                Raise("LengthText");
            };

            player.MediaFailed += delegate
            {
                Trouble = "cannot play this";
            };

            player.MediaEnded += delegate
            {
                Playing = false;
                player.Pause();
            };

            Toggle = new Command(delegate
            {
                if (playing)
                {
                    player.Pause();
                    Playing = false;
                }
                else
                {
                    // Pressing play at the end starts it again from the top.
                    if (length > 0 && player.Position.TotalSeconds >= length - 0.05)
                    {
                        player.Position = TimeSpan.Zero;
                    }

                    player.Play();
                    Playing = true;
                }
            });
        }

        public Command Toggle { get; private set; }

        public bool Playing
        {
            get { return playing; }
            private set
            {
                if (Set(ref playing, value))
                {
                    if (value)
                    {
                        tick.Start();
                    }
                    else
                    {
                        tick.Stop();
                    }
                }
            }
        }

        public string Trouble
        {
            get { return trouble; }
            private set { Set(ref trouble, value); Raise("HasTrouble"); }
        }

        public bool HasTrouble
        {
            get { return !string.IsNullOrEmpty(trouble); }
        }

        public double Length
        {
            get { return length; }
        }

        public double Position
        {
            get { return player.Position.TotalSeconds; }
            set
            {
                player.Position = TimeSpan.FromSeconds(value);
                Raise("PositionText");
            }
        }

        public string PositionText
        {
            get { return Converters.Clock.Format(Position); }
        }

        public string LengthText
        {
            get { return Converters.Clock.Format(length); }
        }

        public void Open(string path)
        {
            Playing = false;
            Trouble = null;
            length = 0;

            if (string.IsNullOrEmpty(path) || !File.Exists(path))
            {
                player.Close();
                Raise("Length");
                Raise("LengthText");
                Raise("Position");
                Raise("PositionText");
                return;
            }

            player.Open(new Uri(path));
        }

        public void Dragging(bool held)
        {
            dragging = held;
        }
    }

    /// <summary>One way of saying it: the voice's own, something written here, or a named style.</summary>
    public class StyleChoice
    {
        public string Label { get; set; }
        public string Instruction { get; set; }

        /// <summary>True for the row whose instruction is whatever is typed on the line below.</summary>
        public bool Written { get; set; }
    }

    /// <summary>text2speech.</summary>
    public class SpeechTaskViewModel : TaskViewModel
    {
        private string text = string.Empty;
        private double speed = 1.0;
        private double temperature = 1.0;
        private string seed = "-1";
        private string instruction = string.Empty;
        private StyleChoice style;
        private Clip reference;

        public SpeechTaskViewModel(MainViewModel main)
            : base(main, TaskKind.Text2Speech)
        {
            Dice = new Command(delegate
            {
                Seed = new Random().Next(1, 65536).ToString(CultureInfo.InvariantCulture);
            });

            ClearReference = new Command(delegate { Reference = null; });
            Player = new PlayerViewModel();
            ResultPlayer = new PlayerViewModel();
            RebuildStyles();
            SelectionChanged();
        }

        /// <summary>The player under the clip that is showing, which is not the one in the settings.</summary>
        public PlayerViewModel ResultPlayer { get; private set; }

        protected override void SelectionChanged()
        {
            if (ResultPlayer != null)
            {
                ResultPlayer.Open(Selected == null ? null : Selected.Path);
            }
        }

        protected override string IconKey
        {
            get { return "IconWaveform"; }
        }

        protected override string BrushKey
        {
            get { return "TaskSpeechBrush"; }
        }

        public Command Dice { get; private set; }

        public Command ClearReference { get; private set; }

        public PlayerViewModel Player { get; private set; }

        public string Text
        {
            get { return text; }
            set
            {
                if (Set(ref text, value))
                {
                    GoChanged();
                }
            }
        }

        /// <summary>
        /// A model that is not a voice says so here, in the footnote under what would be read.
        /// </summary>
        public string NotAVoiceBecause
        {
            get { return Description == null ? null : Description.NotAVoiceBecause; }
        }

        public bool TakesARecording
        {
            get { return Description != null && Description.TakesARecording; }
        }

        public string WhyNotARecording
        {
            get
            {
                if (Description == null || Description.TakesARecording)
                {
                    return null;
                }

                return "This voice cannot be handed a recording: " + Description.WhyNotARecording + ".";
            }
        }

        public Clip Reference
        {
            get { return reference; }
            set
            {
                if (Set(ref reference, value))
                {
                    Raise("HasReference");
                    Raise("DropCaption");
                    Player.Open(value == null ? null : value.Path);
                }
            }
        }

        public bool HasReference
        {
            get { return reference != null; }
        }

        public string DropCaption
        {
            get
            {
                if (reference == null)
                {
                    return "Drop or choose a recording";
                }

                return reference.Readable ? "Drop or choose another" : reference.Trouble;
            }
        }

        public bool HasStyles
        {
            get { return Styles != null && Styles.Count > 2; }
        }

        public ObservableCollection<StyleChoice> Styles { get; private set; }

        public StyleChoice Style
        {
            get { return style; }
            set
            {
                if (!Set(ref style, value) || value == null || value.Written)
                {
                    return;
                }

                // Picking a named style writes its instruction onto the line below, where it can
                // then be edited -- at which point the list is on "As written below".
                Instruction = value.Instruction ?? string.Empty;
            }
        }

        public string Instruction
        {
            get { return instruction; }
            set { Set(ref instruction, value); }
        }

        public double Speed
        {
            get { return speed; }
            set { Set(ref speed, Math.Round(Math.Max(0.25, Math.Min(4, value)), 2)); }
        }

        public double Temperature
        {
            get { return temperature; }
            set { Set(ref temperature, Math.Round(Math.Max(0, Math.Min(2, value)), 2)); }
        }

        public string Seed
        {
            get { return seed; }
            set { Set(ref seed, value); }
        }

        public override bool CanGo
        {
            get
            {
                return !Main.IsQueued(this) && ChosenModel != null && !string.IsNullOrWhiteSpace(Text);
            }
        }

        public void TakeRecording(string path)
        {
            Reference = new Clip(path);
        }

        protected override JobRequest Build()
        {
            return new JobRequest
            {
                Text = Text,
                Speed = Speed,
                Temperature = Temperature,
                Style = string.IsNullOrEmpty(Instruction) ? null : Instruction,
                FromARecording = HasReference,
                SeedText = Seed,
                Steps = 1,

                // A reference recording is taken thirty seconds deep and no further.
                Reference = reference == null || reference.Sound == null
                    ? null
                    : reference.Sound.Head(30),
            };
        }

        protected override void AdoptDescription(ModelDescription model)
        {
            speed = model.Speed;
            temperature = model.Temperature;
            RebuildStyles();

            foreach (var name in new[]
            {
                "Speed", "Temperature", "HasStyles", "TakesARecording", "WhyNotARecording",
                "NotAVoiceBecause",
            })
            {
                Raise(name);
            }
        }

        protected override void ReuseFrom(ResultRecord record)
        {
            var made = record.Made;
            Text = made.Text ?? string.Empty;
            if (made.Speed.HasValue)
            {
                Speed = made.Speed.Value;
            }

            if (made.Temperature.HasValue)
            {
                Temperature = made.Temperature.Value;
            }

            Instruction = made.Style ?? string.Empty;
            Seed = made.Seed.ToString(CultureInfo.InvariantCulture);
        }

        private void RebuildStyles()
        {
            if (Styles == null)
            {
                Styles = new ObservableCollection<StyleChoice>();
                Raise("Styles");
            }

            Styles.Clear();
            Styles.Add(new StyleChoice { Label = "The voice's own way", Instruction = string.Empty });
            Styles.Add(new StyleChoice { Label = "As written below", Written = true });

            if (Description != null && Description.Styles != null)
            {
                foreach (var named in Description.Styles)
                {
                    Styles.Add(new StyleChoice { Label = named.Key, Instruction = named.Value });
                }
            }

            style = Styles[0];
            Raise("Style");
        }
    }

    /// <summary>speech2speech: one recording said again in another voice.</summary>
    public class ConversionTaskViewModel : TaskViewModel
    {
        private Clip from;
        private Clip to;
        private int steps = 30;
        private bool convertStyle;
        private string seed = "-1";

        public ConversionTaskViewModel(MainViewModel main)
            : base(main, TaskKind.Speech2Speech)
        {
            Dice = new Command(delegate
            {
                Seed = new Random().Next(1, 65536).ToString(CultureInfo.InvariantCulture);
            });

            ClearFrom = new Command(delegate { From = null; });
            ClearTo = new Command(delegate { To = null; });
            FromPlayer = new PlayerViewModel();
            ToPlayer = new PlayerViewModel();
            ResultPlayer = new PlayerViewModel();
            SelectionChanged();
        }

        public PlayerViewModel ResultPlayer { get; private set; }

        protected override void SelectionChanged()
        {
            if (ResultPlayer != null)
            {
                ResultPlayer.Open(Selected == null ? null : Selected.Path);
            }
        }

        protected override string IconKey
        {
            get { return "IconConversion"; }
        }

        protected override string BrushKey
        {
            get { return "TaskConversionBrush"; }
        }

        public Command Dice { get; private set; }

        public Command ClearFrom { get; private set; }

        public Command ClearTo { get; private set; }

        public PlayerViewModel FromPlayer { get; private set; }

        public PlayerViewModel ToPlayer { get; private set; }

        public Clip From
        {
            get { return from; }
            set
            {
                if (Set(ref from, value))
                {
                    Raise("HasFrom");
                    Raise("FromCaption");
                    FromPlayer.Open(value == null ? null : value.Path);
                    GoChanged();
                }
            }
        }

        public Clip To
        {
            get { return to; }
            set
            {
                if (Set(ref to, value))
                {
                    Raise("HasTo");
                    Raise("ToCaption");
                    ToPlayer.Open(value == null ? null : value.Path);
                    GoChanged();
                }
            }
        }

        public bool HasFrom
        {
            get { return from != null; }
        }

        public bool HasTo
        {
            get { return to != null; }
        }

        public string FromCaption
        {
            get { return Caption(from); }
        }

        public string ToCaption
        {
            get { return Caption(to); }
        }

        /// <summary>
        /// What the box says. A recording that could not be read says why there rather than
        /// leaving the button dead with nothing to explain it.
        /// </summary>
        private static string Caption(Clip clip)
        {
            if (clip == null)
            {
                return "Drop or choose a recording";
            }

            return clip.Readable ? "Drop or choose another" : clip.Trouble;
        }

        public int Steps
        {
            get { return steps; }
            set { Set(ref steps, Math.Max(1, Math.Min(100, value))); }
        }

        public bool ConvertsStyle
        {
            get { return Description != null && Description.ConvertsStyle; }
        }

        public bool ConvertStyle
        {
            get { return convertStyle; }
            set { Set(ref convertStyle, value); }
        }

        public string Seed
        {
            get { return seed; }
            set { Set(ref seed, value); }
        }

        public override bool CanGo
        {
            get
            {
                return !Main.IsQueued(this) && ChosenModel != null &&
                       from != null && from.Readable && to != null && to.Readable;
            }
        }

        protected override JobRequest Build()
        {
            return new JobRequest
            {
                Steps = Steps,
                ConvertStyle = ConvertsStyle ? (bool?)ConvertStyle : null,
                SeedText = Seed,

                // What is said, up to five minutes of it, and whose voice to say it in.
                Source = from == null || from.Sound == null ? null : from.Sound.Head(300),
                Reference = to == null || to.Sound == null ? null : to.Sound.Head(30),
            };
        }

        protected override void AdoptDescription(ModelDescription model)
        {
            steps = model.Steps;
            Raise("Steps");
            Raise("ConvertsStyle");
        }

        protected override void ReuseFrom(ResultRecord record)
        {
            var made = record.Made;
            if (made.Steps.HasValue)
            {
                Steps = made.Steps.Value;
            }

            ConvertStyle = made.ConvertStyle ?? false;
            Seed = made.Seed.ToString(CultureInfo.InvariantCulture);
        }
    }
}
