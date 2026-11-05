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
// How a rig moves, frame by frame, as Anime2.5DRig moves one (lib/app.js, MIT, Copyright (c) 2026
// hakoniwa): what it is told to do -- where it looks, how open its eyes are, how it
// breathes -- and every vertex of every layer moved by that.
//
// A port of waifu-app/swiftui/Sources/Waifu/RigMotion.swift.
//
// The head turns about the neck and every layer of it moves by how far in front of the face it
// is, which is what makes it look turned rather than slid: front hair more than the face, back
// hair less. The body follows the head a little and leans about its foot. Hair swings in strands,
// each a stiff spring at the root and a soft one at the tip, after the head; an earring or a tail
// sways.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace Libwaifu.Gui.Waifu2d
{
    /// What the rig is told to do, this frame. Angles and the eyes' look are -1 to 1.
    sealed class RigPose
    {
        public float AngleX;
        public float AngleY;
        public float AngleZ;
        public float Body;
        public float EyeX;
        public float EyeY;
        public float Openness = 1;
        /// The iris's scale across and down, from its bounce after a blink.
        public Vector2 Bounce = Vector2.One;
        /// 0 to 1: the chest's breath, and the head's, which lags it.
        public float Breath;
        public float BreathHead;
        /// How soon the open eye gives way to the closed one, 0 to 1.
        public float Ease = 0.3f;
        /// How far the head turns up and down, 1 as the rig turns it.
        public float Nod = 1;
        /// How far the upper body has turned after the head, -1 to 1 as the head's turn is.
        public float BodyTurn;

        // Added for the Windows port: the mouth opening to speech, a smile's corners, the brows
        // up or down, a wink a side, the body twisted, a jump in the air and squashing down, and
        // where a grabbed part is pulled to.
        public float Mouth;
        public float Smile;
        public float Brow;
        public float WinkL;
        public float WinkR;
        public float Twist;
        public float JumpY;
        public float Squash = 1;
        public Vector2 Grab;
    }

    /// What moves on its own.
    sealed class RigSwitches
    {
        public bool Blink = true;
        /// Sways a little, all the time.
        public bool Idle = true;
        /// Looks somewhere else now and then.
        public bool LookAround = true;
        public bool Physics = true;
        public bool FollowMouse = true;
        public bool Wind = true;
        public bool Talk = true;
    }

    /// A spring after a target: Anime2.5DRig's, semi-implicit Euler in steps of at most 1/120 s,
    /// so that it is the same at 10 frames a second as at 240.
    struct Spring
    {
        public float X;
        public float V;

        public void StepTo(float target, float k, float damping, float dt)
        {
            int count = Math.Max(1, (int)Math.Ceiling(dt * 120));
            float h = dt / count;
            for (int i = 0; i < count; i++)
            {
                V += (-k * (X - target) - damping * V) * h;
                X += V * h;
            }
        }
    }

    /// The moving rig's state from one frame to the next.
    sealed class RigMotion
    {
        /// A strand's two springs, and how far each lags its target.
        public sealed class StrandSprings
        {
            public Spring Stiff;
            public Spring Soft;
            public float Phase;
            public float StiffLag;
            public float SoftLag;
        }

        public float Time { get; private set; }
        /// Where the pose is, eased toward where it is told to be.
        public readonly RigPose Pose = new RigPose();
        public readonly Blinker Blinker = new Blinker();
        /// Each part's strands' springs, part by part: empty for one that does not swing.
        public readonly List<StrandSprings[]> Strands;
        Spring bust;
        public float BustLag { get; private set; }

        RigPose look = new RigPose();
        float nextLook;
        readonly Random rng = new Random();

        // Speech level as fed by a mic or made up, 0 to 1, smoothed toward it.
        public float TalkLevel;
        /// How gusty the wind is, 0 to 3.
        public float WindStrength = 1;
        /// Mouth, smile, brows, twist as set by hand, -1 to 1 but the mouth 0 to 1.
        public float MouthManual;
        public float SmileTarget;
        public float BrowTarget;
        public float TwistTarget;
        public float WinkLTarget;
        public float WinkRTarget;
        /// How much the hem of a skirt swings, a spring like the bust's.
        Spring hem;
        public float HemLag { get; private set; }
        float jumpY, jumpV, squash, squashV;
        float shakeT = -1, nodT = -1;
        float gustPhase = 2.1f;
        /// A part being dragged about, -1 while not: its index, and the pull in canvas pixels.
        public int Grabbed = -1;
        Vector2 grabOffset;

        static float Clamp(float v) => Math.Min(Math.Max(v, -1), 1);

        public RigMotion(Rig rig)
        {
            Strands = rig.Parts.Select((part, index) =>
                (part.Swing?.Strands ?? new Strand[0])
                    .Select((s, i) => new StrandSprings { Phase = i * 1.37f + index })
                    .ToArray()
            ).ToList();
        }

        /// Moves on by `dt` seconds. `mouse` is where the pointer is over the canvas, -1 to 1
        /// across it and down it, or null where it is not over it; `open` is as open as the eyes
        /// may be.
        public void Step(double dt, Rig rig, RigSwitches switches, Vector2? mouse, double open,
            double ease, double nod = 1, double follow = 1)
        {
            float fdt = (float)Math.Max(0, Math.Min(dt, 0.05));
            Time += fdt;
            float t = Time;
            var target = new RigPose();

            // Where it is told to look: after the pointer, or wherever it looks next.
            if (switches.FollowMouse && mouse is Vector2 m)
            {
                target.AngleX = Clamp(m.X * 0.9f);
                target.AngleY = Clamp(-m.Y * 0.7f);
                target.EyeX = Clamp(m.X * 1.2f);
                target.EyeY = Clamp(-m.Y * 0.8f);
            }
            if (switches.Idle)
            {
                target.AngleX += 0.13f * MathF.Sin(t * 0.42f) + 0.05f * MathF.Sin(t * 1.13f);
                target.AngleY += 0.08f * MathF.Sin(t * 0.31f + 1.7f);
                target.AngleZ += 0.07f * MathF.Sin(t * 0.23f + 0.5f);
                target.Body += 0.10f * MathF.Sin(t * 0.19f + 2.1f);
            }
            // Anime2.5DRig looks around over the pointer too; here it does not while it follows
            // one, which it would only wander from.
            if (switches.LookAround && !(switches.FollowMouse && mouse != null))
            {
                if (t > nextLook)
                {
                    nextLook = t + 1.4f + Next() * 2.6f;
                    look.AngleX = (Next() * 2 - 1) * 0.55f;
                    look.AngleY = (Next() * 2 - 1) * 0.40f;
                    look.AngleZ = (Next() * 2 - 1) * 0.35f;
                    look.Body = (Next() * 2 - 1) * 0.30f;
                    look.EyeX = (Next() * 2 - 1) * 0.60f;
                    look.EyeY = (Next() * 2 - 1) * 0.35f;
                }
                target.AngleX += look.AngleX;
                target.AngleY += look.AngleY;
                target.AngleZ += look.AngleZ;
                target.Body += look.Body;
                target.EyeX += look.EyeX;
                target.EyeY += look.EyeY;
            }

            // The body leans the way the head turns, as Anime2.5DRig's does after a tracked face.
            target.Body += Clamp(target.AngleX) * 0.25f * (float)follow;

            Blinker.Step(fdt, open, switches.Blink);

            // Eased, 71 ms; the body after the head, over 170.
            float k = 1 - MathF.Exp(-fdt * 14);
            var pose = Pose;
            pose.AngleX += (Clamp(target.AngleX) - pose.AngleX) * k;
            pose.AngleY += (Clamp(target.AngleY) - pose.AngleY) * k;
            pose.AngleZ += (Clamp(target.AngleZ) - pose.AngleZ) * k;
            pose.Body += (Clamp(target.Body) - pose.Body) * (1 - MathF.Exp(-fdt * 6));
            // How far the body has turned after the head: the head's turn, eased the body's way.
            pose.BodyTurn += (Clamp(target.AngleX) * (float)follow - pose.BodyTurn) * (1 - MathF.Exp(-fdt * 6));
            pose.EyeX += (Clamp(target.EyeX) - pose.EyeX) * k;
            pose.EyeY += (Clamp(target.EyeY) - pose.EyeY) * k;
            pose.Openness = (float)Blinker.Openness;
            pose.Bounce = new Vector2((float)Blinker.Bounce.Width, (float)Blinker.Bounce.Height);
            pose.Breath = 0.5f + 0.5f * MathF.Sin(t * 2 * MathF.PI / 3.4f);
            pose.BreathHead = 0.5f + 0.5f * MathF.Sin(t * 2 * MathF.PI / 3.4f - 0.6f);
            pose.Ease = (float)ease;
            pose.Nod = (float)nod;

            // The mouth, as loud as the talk is unless set by hand: fast to open, slow to shut.
            float mouthTarget = Math.Max(TalkLevel, MouthManual);
            float mk = 1 - MathF.Exp(-fdt * (mouthTarget > pose.Mouth ? 30 : 10));
            pose.Mouth += (mouthTarget - pose.Mouth) * mk;
            float ek = 1 - MathF.Exp(-fdt * 10);
            pose.Smile += (SmileTarget - pose.Smile) * ek;
            pose.Brow += (BrowTarget - pose.Brow) * ek;
            pose.Twist += (TwistTarget - pose.Twist) * ek;
            pose.WinkL += (WinkLTarget - pose.WinkL) * ek;
            pose.WinkR += (WinkRTarget - pose.WinkR) * ek;

            // A jump: up off the body pivot, down with gravity, squashing as it lands and
            // stretching back.
            if (jumpY < 0 || jumpV > 0)
            {
                jumpY += jumpV * fdt;
                jumpV += 900 * (float)rig.FaceScale * fdt;
                if (jumpY >= 0) { jumpY = 0; jumpV = 0; squashV -= 4.5f; }
            }
            // The squash, critically not damped: it overshoots, as a landing does.
            squashV += (1 - squash) * 220 * fdt;
            squashV *= MathF.Exp(-fdt * 9);
            squash += squashV * fdt;
            pose.JumpY = jumpY;
            pose.Squash = squash;

            // A shake or a nod told to happen once, decaying away.
            if (shakeT >= 0)
            {
                shakeT += fdt;
                pose.AngleX += MathF.Sin(shakeT * 34) * MathF.Exp(-shakeT * 3.5f) * 0.55f;
                if (shakeT > 1.6f) shakeT = -1;
            }
            if (nodT >= 0)
            {
                nodT += fdt;
                pose.AngleY += MathF.Sin(nodT * 12) * MathF.Exp(-nodT * 2.2f) * 0.5f * pose.Nod;
                if (nodT > 1.4f) nodT = -1;
            }

            // Where a grabbed part is pulled to, easing after the cursor.
            pose.Grab = Grabbed >= 0
                ? pose.Grab + (grabOffset - pose.Grab) * (1 - MathF.Exp(-fdt * 18))
                : pose.Grab * MathF.Exp(-fdt * 10);

            StepSprings(rig, fdt, switches.Wind);
        }

        /// Told to jump: an upward speed, which gravity turns into a hop with a squashy landing.
        public void Jump() { if (jumpY >= 0) jumpV = -300; }
        /// Told to shake or nod once.
        public void Shake() { shakeT = 0; }
        public void NodOnce() { nodT = 0; }
        /// A part grabbed at `offset` canvas pixels from where it was, -1 to let go.
        public void Grab(int index, Vector2 offset)
        {
            // Let go: fling the part's soft springs back with the pull it was under.
            if (index < 0 && Grabbed >= 0)
                foreach (var s in Strands[Grabbed])
                {
                    var soft = s.Soft;
                    soft.V -= Pose.Grab.X * 4;
                    s.Soft = soft;
                }
            Grabbed = index;
            grabOffset = offset;
        }

        float Next() => (float)rng.NextDouble();

        /// Each strand's springs after where the head or the body has gone, and the wind; and the
        /// bust's.
        void StepSprings(Rig rig, float dt, bool wind)
        {
            // The gusts: slow noise over the strength, so the wind comes and goes.
            float gust = Math.Max(0, WindStrength) *
                (0.6f + 0.4f * MathF.Sin(Time * 0.53f + gustPhase) + 0.25f * MathF.Sin(Time * 1.31f + 2));
            var a = rig.Anchors;
            float fs = (float)rig.FaceScale;
            float headDX = (Pose.AngleX * 14 + Pose.AngleZ * 0.07f * (float)(a.NeckPivot.Y - a.FaceCenter.Y)) * fs;
            float bodyDX = Pose.Body * 0.028f * (float)(a.BodyPivot.Y - a.NeckPivot.Y) * 0.35f + Pose.AngleX * 14 * 0.16f * fs;
            for (int index = 0; index < rig.Parts.Count; index++)
            {
                var springs = Strands[index];
                if (springs.Length == 0) continue;
                var part = rig.Parts[index];
                float basis = part.Group == RigGroup.Head ? headDX : bodyDX;
                float windScale = part.Swing?.Kind == Phys.Sway && part.Group != RigGroup.Head ? 2.5f : 1;
                for (int s = 0; s < springs.Length; s++)
                {
                    float phase = springs[s].Phase;
                    float w = wind
                        ? (1.8f * MathF.Sin(Time * 0.8f + phase) + 1.0f * MathF.Sin(Time * 1.9f + phase * 2.3f)) * windScale * gust
                        : 0;
                    float target = basis + w * fs;
                    var stiff = springs[s].Stiff;
                    stiff.StepTo(target, 70, 9, dt);
                    springs[s].Stiff = stiff;
                    springs[s].StiffLag = -(stiff.X - target) * 2.2f;
                    var soft = springs[s].Soft;
                    soft.StepTo(target, 16, 1.3f, dt);
                    springs[s].Soft = soft;
                    springs[s].SoftLag = -(soft.X - target) * 3.0f;
                }
            }
            float bustTarget = (Pose.Breath * 3 - Pose.AngleY * 6 + Pose.Body * 4) * fs;
            bust.StepTo(bustTarget, 140, 4.2f, dt);
            BustLag = -(bust.X - bustTarget) * 3;

            // A skirt's hem, swung like the strands but as one: after the body and the gusts.
            float hemTarget = (bodyDX * 1.4f + (wind ? gust * 2.6f * MathF.Sin(Time * 0.9f + 1.2f) * fs : 0));
            hem.StepTo(hemTarget, 26, 2.0f, dt);
            HemLag = -(hem.X - hemTarget) * 2.4f;
        }

        /// Every part where it is now, and how much of it shows.
        public RigFrame Frame(Rig rig, bool physics) =>
            new RigFrame(
                positions: Enumerable.Range(0, rig.Parts.Count)
                    .Select(i => Deform(rig, i, physics)).ToList(),
                alphas: Enumerable.Range(0, rig.Parts.Count)
                    .Select(i => Alpha(rig, i, Pose)).ToList());

        /// How much of a part shows: an open eye's parts fading into its closed eye as it shuts.
        static float Alpha(Rig rig, int index, RigPose pose)
        {
            var part = rig.Parts[index];
            float opacity = (float)part.Opacity;
            // The long closed eye, for a long blink, is not blinked with.
            if (part.Slot == "eye_close2") return 0;
            if (part.Side is Side side && rig.Eyes.ContainsKey(side))
            {
                // A wink shuts its own eye, as Deform does: fade by the wink-adjusted openness.
                double v = pose.Openness * (1 - (side == Side.Left ? pose.WinkL : pose.WinkR));
                float opened = (float)Smoothing.Smooth((v - (0.10 + pose.Ease * 0.45)) / 0.15);
                switch (part.Role)
                {
                    case Role.EyeWhite:
                    case Role.Irides:
                    case Role.EyeLash: return opacity * opened;
                    case Role.EyeClose: return opacity * (1 - opened);
                }
            }
            return opacity;
        }

        /// Where each vertex of part `index` is now, on the canvas: Anime2.5DRig's deform, in its
        /// order -- the eyes, the head turned, the breath, the chest, the strands, and the body's
        /// lean last.
        public Vector2[] Deform(Rig rig, int index, bool physics)
        {
            var part = rig.Parts[index];
            var e = Pose;
            var a = rig.Anchors;
            float fs = (float)rig.FaceScale;
            var np = new Vector2((float)a.NeckPivot.X, (float)a.NeckPivot.Y);
            var bp = new Vector2((float)a.BodyPivot.X, (float)a.BodyPivot.Y);
            float faceCenterY = (float)a.FaceCenter.Y;
            // The face's surface, which the head turns up and down on: an ellipse over the face's
            // box about its middle, as high in the middle as 0.3 of its width and nothing at its
            // edge.
            var surfaceCenter = new Vector2((float)a.FaceCenter.X, faceCenterY);
            var surfaceRadius = new Vector2(
                Math.Max(1, (float)a.Face.Width / 2), Math.Max(1, (float)a.Face.Height / 2));
            float surfaceHeight = (float)a.Face.Width * 0.3f;
            float surfaceWidth = (float)a.Face.Width;
            float az = e.AngleZ * 0.07f;
            float cz = MathF.Cos(az), sz = MathF.Sin(az);
            float ab = e.Body * 0.028f;
            float cb = MathF.Cos(ab), sb = MathF.Sin(ab);
            var eye = part.Side is Side s && rig.Eyes.TryGetValue(s, out var ea) ? ea : null;
            var eyeRole = eye != null ? part.Role : Role.Other;
            // A wink shuts its own eye over how open both are.
            float v = e.Openness;
            if (part.Side is Side ws)
                v *= 1 - (ws == Side.Left ? e.WinkL : e.WinkR);
            var chest = new Vector2((float)a.Chest.X, (float)a.Chest.Y);
            var reach = new Vector2((float)a.ChestReach.Width, (float)a.ChestReach.Height);
            float neckTop = (float)a.NeckTop;
            float neckBottom = (float)a.NeckBottom;
            bool isTop = part.Slot == "topwear";

            // The mouth's own box, for opening and smiling: its mesh's rest bounds.
            var mouthMin = new Vector2();
            var mouthSize = new Vector2(1, 1);
            var mouthCenter = new Vector2();
            if (part.Slot == "mouth" && part.Mesh.Rest.Length > 0)
            {
                float mnx = float.MaxValue, mny = float.MaxValue, mxx = float.MinValue, mxy = float.MinValue;
                foreach (var p in part.Mesh.Rest)
                {
                    mnx = Math.Min(mnx, p.X); mxx = Math.Max(mxx, p.X);
                    mny = Math.Min(mny, p.Y); mxy = Math.Max(mxy, p.Y);
                }
                mouthMin = new Vector2(mnx, mny);
                mouthSize = new Vector2(Math.Max(1, mxx - mnx), Math.Max(1, mxy - mny));
                mouthCenter = new Vector2((mnx + mxx) / 2, (mny + mxy) / 2);
            }

            // The hem's span, top to bottom of this part's own box.
            float hemTop = float.MaxValue, hemBottom = float.MinValue;
            foreach (var p in part.Mesh.Rest)
            {
                hemTop = Math.Min(hemTop, p.Y);
                hemBottom = Math.Max(hemBottom, p.Y);
            }

            // How the strands swing it.
            var swing = physics ? part.Swing : null;
            var springs = Strands[index];
            float ampK, softK, power;
            if (swing?.Kind == Phys.Sway) { ampK = 1 * 1.4f; softK = 0.8f; power = 1.2f; }
            else if (swing?.Kind == Phys.Hair && swing.Front) { ampK = 2; softK = 0.4f; power = 1.8f; }
            else { ampK = 2; softK = 2; power = 2.1f; }

            var out_ = (Vector2[])part.Mesh.Rest.Clone();
            for (int i = 0; i < out_.Length; i++)
            {
                float x = out_[i].X, y = out_[i].Y;

                // An eye: the white and lashes squashed toward the line it closes onto, the iris
                // bounced, looking, and round until the eye is nearly shut.
                if (eye != null)
                {
                    float closeY = (float)eye.CloseY;
                    switch (eyeRole)
                    {
                        case Role.Irides:
                            float cx = (float)eye.Iris.X, cy = (float)eye.Iris.Y;
                            x = cx + (x - cx) * e.Bounce.X;
                            y = cy + (y - cy) * e.Bounce.Y;
                            // Anime2.5DRig moves it down for a look up; here it goes where it
                            // looks.
                            x += e.EyeX * 11 * fs;
                            y -= e.EyeY * 6 * fs;
                            float shut = (float)Smoothing.Smooth((0.32 - v) / 0.32);
                            y = closeY + (y - closeY) * (1 - 0.80f * shut);
                            break;
                        case Role.EyeWhite:
                        case Role.EyeLash:
                            y = closeY + (y - closeY) * (1 - 0.85f * (1 - v));
                            break;
                        case Role.EyeClose:
                            y -= v * 3;
                            break;
                    }
                }
                if (part.Role == Role.Eyebrow) y += ((1 - v) * 3.5f - e.Brow * 9f) * fs;

                // The mouth: opened to speak, its corners up with a smile -- about its own
                // middle, so it does not wander off the face.
                if (part.Slot == "mouth")
                {
                    float mu = (x - mouthMin.X) / mouthSize.X;   // 0 left, 1 right
                    float mv = (y - mouthMin.Y) / mouthSize.Y;   // 0 top, 1 bottom
                    // Open: the lower lip down and everything a little narrower.
                    float openW = Math.Max(0, mv - 0.25f) / 0.75f;
                    y += e.Mouth * mouthSize.Y * 0.9f * openW * openW;
                    x = mouthCenter.X + (x - mouthCenter.X) * (1 - e.Mouth * 0.25f);
                    // Smile: the corners up and a little out, more as they near the edge.
                    float corner = MathF.Pow(Math.Abs(mu - 0.5f) * 2, 1.6f);
                    y -= e.Smile * mouthSize.Y * (0.55f * corner - 0.10f * (1 - corner));
                    x += e.Smile * mouthSize.X * 0.10f * (mu - 0.5f) * corner;
                }

                // The head turned about the neck: all of the head, a little of the body, the neck
                // more the higher up it.
                float hw = part.Group == RigGroup.Head ? 1 : 0.16f;
                if (part.Slot == "neck")
                    hw = 0.55f * (float)Smoothing.Smooth((neckBottom - y) / Math.Max(1, neckBottom - neckTop));
                if (hw > 0)
                {
                    float rx = x - np.X, ry = y - np.Y;
                    x += (rx * cz - ry * sz - rx) * hw;
                    y += (rx * sz + ry * cz - ry) * hw;
                    float d = part.Depth;
                    x += hw * fs * (e.AngleX * (14 + 40 * (d - 1)) + e.AngleX * (np.Y - y) * 0.028f);
                    if (part.Group == RigGroup.Head)
                    {
                        // Up and down, the head as a ball: every vertex as near as the face's
                        // surface is there, and its layer in front of that, turned about the
                        // face's middle. The middle of the face is nearest, so the eyes, nose and
                        // mouth go farther than its outline, and on an arc; the head is shortened;
                        // and a layer goes as far as it is in front -- the same as Anime2.5DRig's
                        // 30 a unit of depth, which slides the head up and down rather than
                        // turning it.
                        float pitch = e.AngleY * 0.35f * e.Nod;
                        float u = (x - surfaceCenter.X) / surfaceRadius.X;
                        float w2 = (y - surfaceCenter.Y) / surfaceRadius.Y;
                        float z = surfaceHeight * MathF.Sqrt(Math.Max(0, 1 - u * u - w2 * w2))
                            + (d - 1) * 0.3f * surfaceWidth;
                        y = surfaceCenter.Y + (y - surfaceCenter.Y) * MathF.Cos(pitch)
                            - MathF.Sin(pitch) * z - e.AngleY * e.Nod * 9 * fs;
                    }
                    else
                    {
                        y += hw * fs * (-e.AngleY * (9 + 30 * (d - 1))
                            - e.AngleY * (d - 1) * (y - faceCenterY) * 0.05f);
                    }
                }
                // The upper body turned after the head: the more the higher up, nothing at the
                // canvas's foot, and each layer as far as it is in front -- the tie past the top,
                // the arms behind it.
                if (part.Group == RigGroup.Body)
                {
                    float up = (float)Smoothing.Smooth((bp.Y - y) / Math.Max(1, bp.Y - np.Y));
                    x += e.BodyTurn * fs * (6 + 40 * (part.Depth - 0.9f)) * up;
                    // Twisted: the shoulders one way, the hips the other, the middle not at all.
                    x += e.Twist * fs * 26 * (up - 0.5f);
                }

                // The breath: the body rises with it, the head after; the shoulders rise and the
                // top widens.
                y -= (part.Group == RigGroup.Body ? e.Breath * 2.0f : e.BreathHead * 1.6f) * fs;
                if (isTop)
                {
                    if (y < chest.Y)
                        y -= e.Breath * 2.2f * fs * (float)Smoothing.Smooth((chest.Y - y) / (reach.Y * 2));
                    x = np.X + (x - np.X) * (1 + e.Breath * 0.003f);
                    float gx = (x - chest.X) / reach.X;
                    float gy = (y - (chest.Y + 70 * fs)) / reach.Y;
                    y += BustLag * 2.5f * MathF.Exp(-gx * gx - gy * gy);
                }

                // The strands: a stiff root and a soft tip, more of the soft the farther down.
                if (swing != null && springs.Length > 0)
                {
                    float along = swing.Along[i];
                    float u2 = swing.Front ? Math.Min(1, along * 1.6f) : along;
                    float amp = MathF.Pow(u2, power) * ampK;
                    float softMix = Math.Min(1, MathF.Pow(u2, 1.2f) * softK);
                    float dx = 0;
                    int n = springs.Length;
                    for (int si = 0; si < n; si++)
                    {
                        float w = swing.Weights[i * n + si];
                        if (w < 0.001f) continue;
                        dx += w * (springs[si].StiffLag * (1 - softMix) + springs[si].SoftLag * softMix);
                    }
                    x += dx * amp;
                    y += Math.Abs(dx) * amp * 0.12f;
                }

                // A skirt's hem: the lower it is, the more it swings with the body and the wind.
                if (part.Slot == "bottomwear" && physics)
                {
                    float edge = MathF.Pow((float)Smoothing.Smooth(
                        (y - hemTop) / Math.Max(1, hemBottom - hemTop) - 0.45f), 1.6f);
                    x += HemLag * edge;
                }

                // A part being pulled: the tip of its strands goes all the way, the root not.
                if (index == Grabbed)
                {
                    float gw = swing != null && i < swing.Along.Length
                        ? 0.15f + 0.85f * swing.Along[i] : 1f;
                    x += e.Grab.X * gw;
                    y += e.Grab.Y * gw;
                }
                out_[i] = new Vector2(x, y);
            }

            // The body's lean about its foot, which everything leans with.
            if (Math.Abs(ab) > 1e-4f)
            {
                for (int i = 0; i < out_.Length; i++)
                {
                    var r = out_[i] - bp;
                    out_[i] = bp + new Vector2(r.X * cb - r.Y * sb, r.X * sb + r.Y * cb);
                }
            }

            // In the air or squashing down from a jump: the whole rig as one piece.
            if (e.JumpY != 0 || Math.Abs(e.Squash - 1) > 1e-3f)
            {
                for (int i = 0; i < out_.Length; i++)
                {
                    var p = out_[i];
                    float yy = bp.Y + (p.Y - bp.Y) * e.Squash + e.JumpY;
                    float xx = np.X + (p.X - np.X) * (2 - e.Squash);
                    out_[i] = new Vector2(xx, yy);
                }
            }
            return out_;
        }
    }

    /// One frame of a rig: each part's vertices where they are, and how much of it shows.
    sealed class RigFrame
    {
        public readonly List<Vector2[]> Positions;
        public readonly List<float> Alphas;
        public RigFrame(List<Vector2[]> positions, List<float> alphas)
        {
            Positions = positions;
            Alphas = alphas;
        }
    }

    static class MathF
    {
        public const float PI = (float)Math.PI;
        public static float Sin(float v) => (float)Math.Sin(v);
        public static float Cos(float v) => (float)Math.Cos(v);
        public static float Exp(float v) => (float)Math.Exp(v);
        public static float Sqrt(float v) => (float)Math.Sqrt(v);
        public static float Pow(float b, float e) => (float)Math.Pow(b, e);
    }
}
