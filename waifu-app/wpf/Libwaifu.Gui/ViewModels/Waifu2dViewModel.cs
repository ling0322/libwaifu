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
//
using System;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using Libwaifu.Gui.Waifu2d;

namespace Libwaifu.Gui.ViewModels
{
    /// <summary>
    /// The waifu2d page: a layered Photoshop document made to move. Nothing here runs a model --
    /// a .psd out of img2layers, or anything layered the way See-through or Anime2.5DRig names
    /// its layers, is opened and driven about. A port of waifu-app/swiftui's RigPage.
    /// </summary>
    /// <remarks>
    /// The settings the page shows are plain properties; the view's render loop reads them into
    /// the motion every frame, which is the one place the pose is put together.
    /// </remarks>
    public class Waifu2dViewModel : Observable
    {
        private Rig rig;
        private RigMotion motion;
        private string status = "Open a .psd, or drop one on the drawing.";
        private string notes;
        private bool reading;

        // The settings, at the values the standalone window carried.
        private double upDown = 1;
        private double bodyFollows = 1;
        private double eyesOpen = 1;
        private double closesAt = 0.3;
        private double mouthOpen;
        private double smile;
        private double brows;
        private double twist;
        private double gusts = 1;
        private bool followPointer = true;
        private bool sway = true;
        private bool lookAround = true;
        private bool hairSwings = true;
        private bool blinkOnOwn = true;
        private bool autoTalk;
        private bool microphone;
        private bool winkLeft;
        private bool winkRight;
        private bool wind = true;

        public Waifu2dViewModel()
        {
            Open = new Command(delegate
            {
                var dialog = new Microsoft.Win32.OpenFileDialog
                {
                    Filter = "Photoshop documents (*.psd)|*.psd",
                };
                if (dialog.ShowDialog() == true)
                {
                    OpenDocument(dialog.FileName);
                }
            });

            BlinkNow = new Command(delegate { if (motion != null) motion.Blinker.BlinkNow(); });
            Nod = new Command(delegate { if (motion != null) motion.NodOnce(); });
            Shake = new Command(delegate { if (motion != null) motion.Shake(); });
            Jump = new Command(delegate { if (motion != null) motion.Jump(); });
        }

        public string Name
        {
            get { return "waifu2d"; }
        }

        /// <summary>The 20x20 square in the sidebar, the page's own colour.</summary>
        public Geometry Icon
        {
            get { return (Geometry)Application.Current.Resources["IconWaifu"]; }
        }

        public Brush IconBrush
        {
            get { return (Brush)Application.Current.Resources["TaskWaifuBrush"]; }
        }

        // -- the document ------------------------------------------------------------------------

        /// <summary>The rigged document and its pose; the view draws them, null until one opens.</summary>
        internal Rig Rig
        {
            get { return rig; }
        }

        internal RigMotion Motion
        {
            get { return motion; }
        }

        /// <summary>What the toolbar's line says: the file and what the rig made of it.</summary>
        public string Status
        {
            get { return status; }
            private set { Set(ref status, value); }
        }

        /// <summary>What the rig could not do with this document, under the settings.</summary>
        public string Notes
        {
            get { return notes; }
            private set { Set(ref notes, value); }
        }

        /// <summary>True while a document is being read; the button and drop both wait.</summary>
        public bool Reading
        {
            get { return reading; }
            private set { Set(ref reading, value); }
        }

        /// <summary>A document landed; the view is asked to load it, on the UI thread.</summary>
        internal event EventHandler DocumentLoaded;

        /// <summary>
        /// Open a PSD and rig it: read on a worker, reported on the dispatcher's thread, which is
        /// where DocumentLoaded reaches the view.
        /// </summary>
        internal async void OpenDocument(string path)
        {
            if (Reading)
            {
                return;
            }

            Reading = true;
            Status = "Reading " + Path.GetFileName(path) + "...";
            try
            {
                var made = await Task.Run(() => Rig.Build(PsdDocument.Read(File.ReadAllBytes(path))));
                rig = made;
                motion = new RigMotion(made);
                Status = string.Format(
                    "{0}: {1} layers, {2} x {3}; {4} of 2 eyes blink.",
                    Path.GetFileName(path), made.Parts.Count, made.Width, made.Height,
                    made.Eyes.Count);
                Notes = string.Join("\n", made.Notes);
                Raise("Rig");
                Raise("Motion");
                var loaded = DocumentLoaded;
                if (loaded != null)
                {
                    loaded(this, EventArgs.Empty);
                }
            }
            catch (Exception error)
            {
                Engine.Log.Write("warn", "waifu2d", path + ": " + error.Message);
                Status = Path.GetFileName(path) + ": " + error.Message;
            }
            finally
            {
                Reading = false;
            }
        }

        public Command Open { get; private set; }

        // -- motion --------------------------------------------------------------------------------

        public bool FollowPointer
        {
            get { return followPointer; }
            set { Set(ref followPointer, value); }
        }

        public bool Sway
        {
            get { return sway; }
            set { Set(ref sway, value); }
        }

        public bool LookAround
        {
            get { return lookAround; }
            set { Set(ref lookAround, value); }
        }

        public bool HairSwings
        {
            get { return hairSwings; }
            set { Set(ref hairSwings, value); }
        }

        /// <summary>How far the head tips up and down, 0 to 2.</summary>
        public double UpDown
        {
            get { return upDown; }
            set { Set(ref upDown, value); }
        }

        /// <summary>How far the body turns after the head, 0 to 2.</summary>
        public double BodyFollows
        {
            get { return bodyFollows; }
            set { Set(ref bodyFollows, value); }
        }

        // -- blink -----------------------------------------------------------------------------------

        public bool BlinkOnOwn
        {
            get { return blinkOnOwn; }
            set { Set(ref blinkOnOwn, value); }
        }

        /// <summary>How open the eyes are held, 0 to 1.</summary>
        public double EyesOpen
        {
            get { return eyesOpen; }
            set { Set(ref eyesOpen, value); }
        }

        /// <summary>How soon the open eye gives way to the closed one, 0 to 1.</summary>
        public double ClosesAt
        {
            get { return closesAt; }
            set { Set(ref closesAt, value); }
        }

        // -- talk -----------------------------------------------------------------------------------

        /// <summary>Made-up babble: the mouth opens and shuts as if speaking.</summary>
        public bool AutoTalk
        {
            get { return autoTalk; }
            set { Set(ref autoTalk, value); }
        }

        /// <summary>The mic's loudness moves the mouth; the view polls it.</summary>
        public bool Microphone
        {
            get { return microphone; }
            set { Set(ref microphone, value); }
        }

        /// <summary>The mouth opened by hand, 0 to 1, whichever of this and talking is louder.</summary>
        public double MouthOpen
        {
            get { return mouthOpen; }
            set { Set(ref mouthOpen, value); }
        }

        // -- expression -----------------------------------------------------------------------------

        public double Smile
        {
            get { return smile; }
            set { Set(ref smile, value); }
        }

        public double Brows
        {
            get { return brows; }
            set { Set(ref brows, value); }
        }

        public bool WinkLeft
        {
            get { return winkLeft; }
            set { Set(ref winkLeft, value); }
        }

        public bool WinkRight
        {
            get { return winkRight; }
            set { Set(ref winkRight, value); }
        }

        public double Twist
        {
            get { return twist; }
            set { Set(ref twist, value); }
        }

        public Command BlinkNow { get; private set; }

        public Command Nod { get; private set; }

        public Command Shake { get; private set; }

        public Command Jump { get; private set; }

        // -- world ---------------------------------------------------------------------------------

        public bool Wind
        {
            get { return wind; }
            set { Set(ref wind, value); }
        }

        /// <summary>How hard the gusts push the hair and the hem, 0 to 3.</summary>
        public double WindGusts
        {
            get { return gusts; }
            set { Set(ref gusts, value); }
        }
    }
}
