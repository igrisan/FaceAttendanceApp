namespace FaceAttendanceApp.Helpers
{
    /// <summary>
    /// Smooths per-frame liveness and match scores over a short rolling window.
    ///
    /// Why: your logs show liveness confidence swinging 99.9% -> 98.4% -> 61.4%
    /// across consecutive frames of the SAME stationary person, purely from
    /// motion blur / exposure noise on individual frames. Trusting a single frame
    /// means a single bad frame can wrongly reject a real worker or (less likely
    /// but possible) let a borderline spoof attempt slip through on a lucky frame.
    ///
    /// This buffers the last N frames' results for the currently-tracked face and
    /// only commits an attendance record once a decision is stable across the
    /// window — same idea as debouncing a noisy sensor.
    /// </summary>
    public class TemporalVoter
    {
        private readonly int _windowSize;
        private readonly double _liveConfidenceThreshold;
        private readonly double _requiredLiveFraction;

        private readonly Queue<FrameResult> _window = new();

        public TemporalVoter(int windowSize = 3, double liveConfidenceThreshold = 0.85, double requiredLiveFraction = 0.66)
        {
            _windowSize = windowSize;
            _liveConfidenceThreshold = liveConfidenceThreshold;
            _requiredLiveFraction = requiredLiveFraction;
        }

        public record FrameResult(bool IsLive, double LiveConfidence, string? WorkerId, float MatchScore);

        /// <summary>
        /// Feed one frame's result in. Returns the current voting state.
        /// Call Reset() whenever face tracking is lost (no face detected, or a
        /// different person's face appears) so stale frames from a previous
        /// person never blend into a new decision.
        /// </summary>
        public VoteResult AddFrame(FrameResult frame)
        {
            _window.Enqueue(frame);
            while (_window.Count > _windowSize)
                _window.Dequeue();

            if (_window.Count < _windowSize)
            {
                return new VoteResult(Decided: false, IsLive: false, WorkerId: null, AvgMatchScore: 0, AvgLiveConfidence: 0);
            }

            int liveFrames = _window.Count(f => f.IsLive && f.LiveConfidence >= _liveConfidenceThreshold);
            double liveFraction = (double)liveFrames / _window.Count;
            bool isLive = liveFraction >= _requiredLiveFraction;

            // Worker ID must be consistent across the window too — if the matcher
            // flip-flopped between two different workers, that's not a stable
            // decision either, treat as undecided.
            var distinctWorkers = _window.Select(f => f.WorkerId).Distinct().ToList();
            string? consensusWorker = distinctWorkers.Count == 1 ? distinctWorkers[0] : null;

            double avgMatch = _window.Average(f => f.MatchScore);
            double avgLiveConf = _window.Average(f => f.LiveConfidence);

            bool decided = isLive && consensusWorker != null;

            return new VoteResult(decided, isLive, consensusWorker, avgMatch, avgLiveConf);
        }

        public void Reset() => _window.Clear();

        public record VoteResult(bool Decided, bool IsLive, string? WorkerId, double AvgMatchScore, double AvgLiveConfidence);
    }
}