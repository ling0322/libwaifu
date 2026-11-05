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
using System.Numerics;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Libwaifu.Gui.ViewModels;
using Libwaifu.Gui.Waifu2d;
using SkiaSharp;

namespace Libwaifu.Gui.Views
{
    /// <summary>
    /// The view half of waifu2d: the mesh renderer, the frame the screen composes, and the
    /// pointer, drag and microphone the pose is fed. The document and its settings are the
    /// view model's; which of them the pose reads is copied in every frame.
    /// </summary>
    public partial class Waifu2dView : UserControl
    {
        private readonly RigRenderer renderer = new RigRenderer();
        private readonly RigSwitches switches = new RigSwitches();
        private WriteableBitmap bitmap;
        private Vector2? pointer;
        private DateTime lastFrame = DateTime.MinValue;
        private MicLevel mic;
        private int dragging = -1;
        private Vector2 grabStart, grabNow;

        public Waifu2dView()
        {
            InitializeComponent();
            Loaded += OnLoaded;
            Unloaded += OnUnloaded;
        }

        private Waifu2dViewModel Model
        {
            get { return DataContext as Waifu2dViewModel; }
        }

        private void OnLoaded(object sender, RoutedEventArgs e)
        {
            var model = Model;
            if (model != null)
            {
                model.DocumentLoaded += OnDocumentLoaded;
                if (model.Rig != null)
                {
                    renderer.Load(model.Rig);
                }
            }

            CompositionTarget.Rendering += Frame;
            // Space jumps, whichever control holds the focus while this page is up.
            var window = Window.GetWindow(this);
            if (window != null)
            {
                window.PreviewKeyDown += OnSpaceKey;
            }
        }

        private void OnUnloaded(object sender, RoutedEventArgs e)
        {
            CompositionTarget.Rendering -= Frame;
            var model = Model;
            if (model != null)
            {
                model.DocumentLoaded -= OnDocumentLoaded;
            }

            var window = Window.GetWindow(this);
            if (window != null)
            {
                window.PreviewKeyDown -= OnSpaceKey;
            }

            renderer.Dispose();
            if (mic != null)
            {
                mic.Dispose();
                mic = null;
            }
        }

        private void OnSpaceKey(object sender, KeyEventArgs e)
        {
            var model = Model;
            // A space pressed on a control -- a box, a switch, a button -- is that control's,
            // not a jump.
            if (e.Key == Key.Space && IsVisible && !(e.OriginalSource is Control) &&
                model != null && model.Motion != null)
            {
                model.Motion.Jump();
                e.Handled = true;
            }
        }

        private void OnDocumentLoaded(object sender, EventArgs e)
        {
            renderer.Load(Model.Rig);
        }

        // -- the pointer and the mouse ------------------------------------------------------------

        private void OnMouseMove(object sender, MouseEventArgs e)
        {
            pointer = PointerAt(e.GetPosition(Stage));
            if (dragging >= 0)
            {
                grabNow = CanvasAt(e.GetPosition(Stage));
            }
        }

        private void OnMouseLeave(object sender, MouseEventArgs e)
        {
            pointer = null;
        }

        private void OnMouseDown(object sender, MouseButtonEventArgs e)
        {
            var model = Model;
            if (model == null || model.Motion == null)
            {
                return;
            }

            var at = CanvasAt(e.GetPosition(Stage));
            int hit = PickPart(at);
            if (hit < 0)
            {
                return;
            }

            dragging = hit;
            grabStart = grabNow = at;
            model.Motion.Grab(hit, Vector2.Zero);
            Stage.CaptureMouse();
        }

        private void OnMouseUp(object sender, MouseButtonEventArgs e)
        {
            var model = Model;
            if (dragging < 0)
            {
                return;
            }

            dragging = -1;
            if (model != null && model.Motion != null)
            {
                model.Motion.Grab(-1, Vector2.Zero);
            }

            Stage.ReleaseMouseCapture();
        }

        private void OnDrop(object sender, DragEventArgs e)
        {
            var files = e.Data.GetData(DataFormats.FileDrop) as string[];
            var model = Model;
            if (files != null && files.Length > 0 && model != null)
            {
                model.OpenDocument(files[0]);
            }
        }

        private void OnDragOver(object sender, DragEventArgs e)
        {
            e.Effects = e.Data.GetDataPresent(DataFormats.FileDrop)
                ? DragDropEffects.Copy
                : DragDropEffects.None;
            e.Handled = true;
        }

        /// Where the pointer is over the drawing in the stage: -1 to 1 across it and down it, as
        /// far as 1.5 beyond it, or null when it is out.
        private Vector2? PointerAt(System.Windows.Point p)
        {
            var model = Model;
            if (model == null || model.Rig == null || bitmap == null)
            {
                return null;
            }

            var fitted = RigRenderer.Fit(
                model.Rig.Width, model.Rig.Height, (float)Stage.ActualWidth, (float)Stage.ActualHeight);
            if (fitted.Width <= 0 || fitted.Height <= 0)
            {
                return null;
            }

            double x = (p.X - fitted.Left) / fitted.Width * 2 - 1;
            // Down the drawing: the view's y is up.
            double y = (1 - (p.Y - fitted.Top) / fitted.Height) * 2 - 1;
            if (x < -1.5 || x > 1.5 || y < -1.5 || y > 1.5)
            {
                return null;
            }

            return new Vector2(
                (float)Math.Min(Math.Max(x, -1.5), 1.5),
                (float)Math.Min(Math.Max(y, -1.5), 1.5));
        }

        /// A stage point in canvas pixels, the same space the rig's vertices live in.
        private Vector2 CanvasAt(System.Windows.Point p)
        {
            var rig = Model.Rig;
            var fitted = RigRenderer.Fit(
                rig.Width, rig.Height, (float)Stage.ActualWidth, (float)Stage.ActualHeight);
            return new Vector2(
                (float)((p.X - fitted.Left) / Math.Max(1, fitted.Width) * rig.Width),
                (float)((p.Y - fitted.Top) / Math.Max(1, fitted.Height) * rig.Height));
        }

        /// The part under a canvas point, topmost first: a swinging one if there is one under it.
        private int PickPart(Vector2 at)
        {
            var rig = Model.Rig;
            int any = -1;
            for (int i = rig.Parts.Count - 1; i >= 0; i--)
            {
                var part = rig.Parts[i];
                if (part.Mesh.Rest.Length == 0)
                {
                    continue;
                }

                float mnx = float.MaxValue, mny = float.MaxValue;
                float mxx = float.MinValue, mxy = float.MinValue;
                foreach (var p in part.Mesh.Rest)
                {
                    mnx = Math.Min(mnx, p.X);
                    mxx = Math.Max(mxx, p.X);
                    mny = Math.Min(mny, p.Y);
                    mxy = Math.Max(mxy, p.Y);
                }

                if (at.X < mnx || at.X > mxx || at.Y < mny || at.Y > mxy)
                {
                    continue;
                }

                if (part.Swing != null || part.Slot == "bottomwear")
                {
                    return i;
                }

                if (any < 0)
                {
                    any = i;
                }
            }

            return any;
        }

        // -- the frame ------------------------------------------------------------------------------

        /// Moves the rig on and draws it, each frame the screen composes.
        private void Frame(object sender, EventArgs e)
        {
            var now = DateTime.UtcNow;
            double dt = lastFrame == DateTime.MinValue ? 0 : (now - lastFrame).TotalSeconds;
            lastFrame = now;

            var model = Model;
            if (model == null || model.Rig == null || model.Motion == null || model.Reading)
            {
                return;
            }

            // Draw at the stage's pixel size (DIP aware).
            var dpi = VisualTreeHelper.GetDpi(this);
            int w = Math.Max(1, (int)(Stage.ActualWidth * dpi.DpiScaleX));
            int h = Math.Max(1, (int)(Stage.ActualHeight * dpi.DpiScaleY));
            if (bitmap == null || bitmap.PixelWidth != w || bitmap.PixelHeight != h)
            {
                bitmap = new WriteableBitmap(w, h, 96 * dpi.DpiScaleX, 96 * dpi.DpiScaleY,
                    PixelFormats.Pbgra32, null);
                Picture.Source = bitmap;
            }

            // What the page's settings leave on: the switches, the mouth, the face, the wind.
            switches.FollowMouse = model.FollowPointer;
            switches.Idle = model.Sway;
            switches.LookAround = model.LookAround;
            switches.Physics = model.HairSwings;
            switches.Blink = model.BlinkOnOwn;
            switches.Wind = model.Wind;

            var motion = model.Motion;
            motion.MouthManual = (float)model.MouthOpen;
            motion.SmileTarget = (float)model.Smile;
            motion.BrowTarget = (float)model.Brows;
            motion.TwistTarget = (float)model.Twist;
            motion.WinkLTarget = model.WinkLeft ? 1 : 0;
            motion.WinkRTarget = model.WinkRight ? 1 : 0;
            motion.WindStrength = (float)model.WindGusts;

            // The talk level: the mic's loudness while it is on, or made-up babble, or both.
            // The device is opened by the toggle and handed back the moment it turns off.
            float talk = 0;
            if (model.Microphone)
            {
                if (mic == null)
                {
                    try { mic = new MicLevel(); } catch { }
                }

                if (mic != null && mic.Open)
                {
                    mic.Poll();
                    talk = mic.Level;
                }
            }
            else if (mic != null)
            {
                mic.Dispose();
                mic = null;
            }

            if (model.AutoTalk)
            {
                double bt = DateTime.UtcNow.TimeOfDay.TotalSeconds;
                // Bursts of babble with gaps, as speech comes and goes.
                double gate = Math.Max(0, Math.Sin(bt * 0.9) * 0.6 + Math.Sin(bt * 0.31 + 1) * 0.5);
                double babble = Math.Abs(Math.Sin(bt * 21) * 0.55 + Math.Sin(bt * 33 + 0.7) * 0.45);
                talk = Math.Max(talk, (float)(gate * babble));
            }

            motion.TalkLevel = talk;
            if (dragging >= 0)
            {
                motion.Grab(dragging, grabNow - grabStart);
            }

            motion.Step(dt, model.Rig, switches, pointer,
                model.EyesOpen, model.ClosesAt, model.UpDown, model.BodyFollows);
            var frame = motion.Frame(model.Rig, switches.Physics);

            bitmap.Lock();
            var info = new SKImageInfo(w, h, SKColorType.Bgra8888, SKAlphaType.Premul);
            using (var surface = SKSurface.Create(info, bitmap.BackBuffer, bitmap.BackBufferStride))
            {
                renderer.Encode(surface.Canvas, frame, w, h, new SKColor(0, 0, 0, 0));
            }

            bitmap.AddDirtyRect(new Int32Rect(0, 0, w, h));
            bitmap.Unlock();
        }
    }
}
