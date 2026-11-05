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
// The microphone's loudness for the mouth to move to, through winmm's waveIn: one buffer of
// 16-bit samples polled a frame at a time, no callback. The mouth opens as loud as `Level`,
// 0 quiet to 1 loud, smoothed a little here so it does not chatter.

using System;
using System.Runtime.InteropServices;

namespace Libwaifu.Gui.Waifu2d
{
    sealed class MicLevel : IDisposable
    {
        const int WAVE_MAPPER = -1;
        const int CALLBACK_NULL = 0;
        const int WHDR_DONE = 1;
        const int WAVE_FORMAT_PCM = 1;
        const int MMSYSERR_NOERROR = 0;

        [DllImport("winmm.dll")] static extern int waveInOpen(out IntPtr h, int dev,
            ref WaveFormatEx fmt, IntPtr cb, IntPtr inst, int flags);
        [DllImport("winmm.dll")] static extern int waveInStart(IntPtr h);
        [DllImport("winmm.dll")] static extern int waveInStop(IntPtr h);
        [DllImport("winmm.dll")] static extern int waveInReset(IntPtr h);
        [DllImport("winmm.dll")] static extern int waveInClose(IntPtr h);
        [DllImport("winmm.dll")] static extern int waveInPrepareHeader(IntPtr h, IntPtr hdr, int size);
        [DllImport("winmm.dll")] static extern int waveInUnprepareHeader(IntPtr h, IntPtr hdr, int size);
        [DllImport("winmm.dll")] static extern int waveInAddBuffer(IntPtr h, IntPtr hdr, int size);

        [StructLayout(LayoutKind.Sequential)]
        struct WaveFormatEx
        {
            public ushort wFormatTag, nChannels;
            public uint nSamplesPerSec, nAvgBytesPerSec;
            public ushort nBlockAlign, wBitsPerSample, cbSize;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct WaveHdr
        {
            public IntPtr lpData;
            public uint dwBufferLength, dwBytesRecorded;
            public IntPtr dwUser;
            public uint dwFlags, dwLoops;
            public IntPtr lpNext, reserved;
        }

        IntPtr hIn, hdr, buf;
        int flagsOff, dataOff;
        const int SampleRate = 11025;
        const int BufferBytes = 8192;   // 4096 samples, about 0.37 s
        short[] samples = new short[BufferBytes / 2];

        /// As loud as the mic is, 0 to 1.
        public float Level { get; private set; }
        /// Whether a mic was found and is listening.
        public bool Open { get; private set; }

        public MicLevel()
        {
            var fmt = new WaveFormatEx
            {
                wFormatTag = WAVE_FORMAT_PCM,
                nChannels = 1,
                nSamplesPerSec = SampleRate,
                nAvgBytesPerSec = SampleRate * 2,
                nBlockAlign = 2,
                wBitsPerSample = 16,
                cbSize = 0,
            };
            if (waveInOpen(out hIn, WAVE_MAPPER, ref fmt, IntPtr.Zero, IntPtr.Zero, CALLBACK_NULL)
                != MMSYSERR_NOERROR) return;

            buf = Marshal.AllocHGlobal(BufferBytes);
            int hdrSize = Marshal.SizeOf<WaveHdr>();
            hdr = Marshal.AllocHGlobal(hdrSize);
            for (int i = 0; i < hdrSize; i++) Marshal.WriteByte(hdr, i, 0);
            flagsOff = (int)Marshal.OffsetOf<WaveHdr>("dwFlags");
            dataOff = (int)Marshal.OffsetOf<WaveHdr>("lpData");
            Marshal.WriteIntPtr(hdr, dataOff, buf);
            Marshal.WriteInt32(hdr, (int)Marshal.OffsetOf<WaveHdr>("dwBufferLength"), BufferBytes);
            if (waveInPrepareHeader(hIn, hdr, hdrSize) != MMSYSERR_NOERROR
                || waveInAddBuffer(hIn, hdr, hdrSize) != MMSYSERR_NOERROR
                || waveInStart(hIn) != MMSYSERR_NOERROR)
            {
                Dispose();
                return;
            }
            Open = true;
        }

        /// Called once a frame: reads a finished buffer's loudness and queues it again.
        public void Poll()
        {
            if (!Open) return;
            if ((Marshal.ReadInt32(hdr, flagsOff) & WHDR_DONE) == 0) return;
            Marshal.Copy(buf, samples, 0, samples.Length);
            waveInUnprepareHeader(hIn, hdr, Marshal.SizeOf<WaveHdr>());
            Marshal.WriteIntPtr(hdr, dataOff, buf);
            Marshal.WriteInt32(hdr, flagsOff, 0);
            waveInPrepareHeader(hIn, hdr, Marshal.SizeOf<WaveHdr>());
            waveInAddBuffer(hIn, hdr, Marshal.SizeOf<WaveHdr>());

            double sum = 0;
            foreach (var s in samples) sum += (double)s * s;
            float rms = (float)Math.Sqrt(sum / samples.Length) / 32768f;
            // Speech sits about 0.01 to 0.2: gain it up, and keep a little noise floor out.
            float level = Math.Min(1, Math.Max(0, (rms - 0.012f) * 8));
            Level += (level - Level) * (level > Level ? 0.5f : 0.12f);
        }

        public void Dispose()
        {
            Open = false;
            if (hIn != IntPtr.Zero)
            {
                waveInStop(hIn);
                waveInReset(hIn);
                if (hdr != IntPtr.Zero)
                    waveInUnprepareHeader(hIn, hdr, Marshal.SizeOf<WaveHdr>());
                waveInClose(hIn);
                hIn = IntPtr.Zero;
            }
            if (buf != IntPtr.Zero) { Marshal.FreeHGlobal(buf); buf = IntPtr.Zero; }
            if (hdr != IntPtr.Zero) { Marshal.FreeHGlobal(hdr); hdr = IntPtr.Zero; }
        }
    }
}
