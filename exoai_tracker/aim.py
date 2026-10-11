"""Proportional aim driven by fresh detections. No prediction, no lead.

Detection lands in ~3ms, so there is nothing worth predicting. Two things did
have to be modelled properly, because getting them wrong is what made the
crosshair orbit its target:

  * One mouse count is not one screen pixel. The ratio depends on in-game
    sensitivity, so it is calibrated online from what we send versus how far
    the target actually slides.
  * A correction smaller than one count cannot be sent. Rounding those up to a
    full count every detection injects hundreds of counts per second, and since
    X and Y round independently the result is a circle. Remainders are carried
    instead.
"""
from __future__ import annotations

import math
import time
from collections import deque

from exoai_tracker.infer import Det

# Kinds worth shooting at.
HOSTILE = frozenset({"enemy", "knocked"})


def aim_point(d: Det, offset_y: float) -> tuple[float, float]:
    """offset_y 0.5 = box centre, lower = higher on the body."""
    return d.x, d.y + (0.5 - offset_y) * d.h


class DetectionFilter:
    """Filters and smooths detections for more stable tracking."""
    
    def __init__(
        self,
        min_size: int = 15,
        max_size: int = 400,
        min_aspect: float = 0.8,
        max_aspect: float = 4.0,
        filter_own: bool = True,
        smoothing: int = 1,
        capture_size: int = 480,
    ) -> None:
        self.min_size = min_size
        self.max_size = max_size
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.filter_own = filter_own
        self.smoothing = max(1, smoothing)
        self.capture_size = capture_size
        # Only the selected target is smoothed. Track it by spatial continuity
        # instead of a collision-prone position hash.
        self._history: list[tuple[float, float, float]] = []
        self._history_cls: int | None = None
    
    def configure(
        self,
        min_size: int | None = None,
        max_size: int | None = None,
        min_aspect: float | None = None,
        max_aspect: float | None = None,
        filter_own: bool | None = None,
        smoothing: int | None = None,
        capture_size: int | None = None,
    ) -> None:
        if min_size is not None:
            self.min_size = min_size
        if max_size is not None:
            self.max_size = max_size
        if min_aspect is not None:
            self.min_aspect = min_aspect
        if max_aspect is not None:
            self.max_aspect = max_aspect
        if filter_own is not None:
            self.filter_own = filter_own
        if smoothing is not None:
            self.smoothing = max(1, smoothing)
        if capture_size is not None:
            self.capture_size = capture_size
    
    def filter(self, dets: list[Det], cx: float, cy: float) -> list[Det]:
        """Filter detections based on size, aspect ratio, and position."""
        filtered: list[Det] = []
        
        for d in dets:
            # Only target hostile
            if d.kind not in HOSTILE:
                continue
            
            # Size filter
            size = max(d.w, d.h)
            if size < self.min_size or size > self.max_size:
                continue
            
            # Aspect ratio filter (height / width)
            if d.w > 0:
                aspect = d.h / d.w
                if aspect < self.min_aspect or aspect > self.max_aspect:
                    continue
            
            # Own player filter: bottom corners of capture region
            if self.filter_own:
                # Detection coords are screen-relative, convert to capture-relative
                rel_x = d.x - cx + self.capture_size / 2
                rel_y = d.y - cy + self.capture_size / 2
                # Bottom 20% and left/right 15%
                if rel_y > self.capture_size * 0.80:
                    if rel_x < self.capture_size * 0.15 or rel_x > self.capture_size * 0.85:
                        continue
            
            filtered.append(d)
        
        return filtered
    
    def smooth(self, det: Det, now: float) -> Det:
        """Apply temporal smoothing to detection position."""
        if self.smoothing <= 1:
            return det
        
        hist = self._history
        if hist:
            last_x, last_y, _ = hist[-1]
            continuity = max(det.w, det.h) * 2.0
            if (
                self._history_cls != det.cls
                or math.hypot(det.x - last_x, det.y - last_y) > continuity
            ):
                hist.clear()
        self._history_cls = det.cls
        hist.append((det.x, det.y, now))
        
        # Keep only recent entries
        cutoff = now - 0.1  # 100ms window
        hist[:] = [(x, y, t) for x, y, t in hist if t > cutoff][-self.smoothing:]
        
        if len(hist) < 2:
            return det
        
        # Average position
        avg_x = sum(x for x, y, t in hist) / len(hist)
        avg_y = sum(y for x, y, t in hist) / len(hist)
        
        return Det(
            x=avg_x, y=avg_y, w=det.w, h=det.h, conf=det.conf,
            cls=det.cls, name=det.name, kind=det.kind,
        )
    
    def predict(self, det: Det, frames: int, now: float) -> Det:
        """Predict future position based on velocity."""
        if frames <= 0:
            return det
        
        hist = self._history
        
        if len(hist) < 2:
            return det
        
        # Calculate velocity from last two positions
        x1, y1, t1 = hist[-2]
        x2, y2, t2 = hist[-1]
        dt = t2 - t1
        
        if dt < 0.001:
            return det
        
        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt
        
        # Predict by the measured frame interval, capped to avoid runaway lead
        # after a stall or breakpoint.
        predict_dt = min(0.05, frames * dt)
        pred_x = det.x + vx * predict_dt
        pred_y = det.y + vy * predict_dt
        
        return Det(
            x=pred_x, y=pred_y, w=det.w, h=det.h, conf=det.conf,
            cls=det.cls, name=det.name, kind=det.kind,
        )


class TargetSelector:
    """Selects best target with configurable priority and lock-on."""
    
    def __init__(
        self,
        priority: str = "closest",
        lock_time: float = 0.08,
        sticky_time: float = 0.15,
    ) -> None:
        self.priority = priority
        self.lock_time = lock_time
        self.sticky_time = sticky_time
        self._locked: Det | None = None
        self._lock_start = 0.0
        self._last_seen = 0.0
    
    def configure(
        self,
        priority: str | None = None,
        lock_time: float | None = None,
        sticky_time: float | None = None,
    ) -> None:
        if priority is not None:
            self.priority = priority
        if lock_time is not None:
            self.lock_time = lock_time
        if sticky_time is not None:
            self.sticky_time = sticky_time
    
    def select(
        self,
        dets: list[Det],
        cx: float,
        cy: float,
        fov: float,
        offset_y: float,
        now: float,
    ) -> Det | None:
        """Select best target from filtered detections."""
        if not dets:
            # Sticky aim: return last target if recently seen
            if self._locked and (now - self._last_seen) < self.sticky_time:
                return self._aim_det(self._locked, offset_y)
            self._locked = None
            return None
        
        fov_sq = fov * fov
        candidates: list[tuple[Det, float, float, float]] = []  # (det, dist, conf, size)
        
        for d in dets:
            ax, ay = aim_point(d, offset_y)
            dist_sq = (ax - cx) ** 2 + (ay - cy) ** 2
            if dist_sq > fov_sq:
                continue
            dist = math.sqrt(dist_sq)
            size = d.w * d.h
            candidates.append((d, dist, d.conf, size))
        
        if not candidates:
            if self._locked and (now - self._last_seen) < self.sticky_time:
                return self._aim_det(self._locked, offset_y)
            self._locked = None
            return None
        
        # Check if current locked target is still valid
        if self._locked and (now - self._lock_start) < self.lock_time:
            for d, dist, conf, size in candidates:
                if self._is_same_target(d, self._locked):
                    self._locked = d
                    self._last_seen = now
                    return self._aim_det(d, offset_y)
        
        # Select based on priority
        if self.priority == "closest":
            candidates.sort(key=lambda x: x[1])
        elif self.priority == "highest_conf":
            candidates.sort(key=lambda x: -x[2])
        elif self.priority == "largest":
            candidates.sort(key=lambda x: -x[3])
        elif self.priority == "center":
            # Prefer targets near crosshair with high confidence
            candidates.sort(key=lambda x: x[1] * 0.7 - x[2] * 50)
        
        best = candidates[0][0]
        self._locked = best
        self._lock_start = now
        self._last_seen = now
        
        return self._aim_det(best, offset_y)

    @staticmethod
    def _aim_det(det: Det, offset_y: float) -> Det:
        ax, ay = aim_point(det, offset_y)
        return Det(
            x=ax, y=ay, w=det.w, h=det.h, conf=det.conf,
            cls=det.cls, name=det.name, kind=det.kind,
        )
    
    def _is_same_target(self, a: Det, b: Det) -> bool:
        """Check if two detections are likely the same target."""
        dist = math.hypot(a.x - b.x, a.y - b.y)
        return dist < max(a.w, a.h, b.w, b.h) * 1.5
    
    def reset(self) -> None:
        self._locked = None
        self._lock_start = 0.0
        self._last_seen = 0.0


def pick_target(
    dets: list[Det],
    cx: float,
    cy: float,
    *,
    fov: float,
    offset_y: float = 0.5,
    lock: Det | None = None,
) -> Det | None:
    """Legacy function for backwards compatibility."""
    fov_sq = fov * fov
    best: Det | None = None
    best_d = fov_sq

    for d in dets:
        if d.kind not in HOSTILE:
            continue
        ax, ay = aim_point(d, offset_y)
        dist = (ax - cx) ** 2 + (ay - cy) ** 2
        if dist < best_d:
            best_d = dist
            best = d

    if best is None:
        return None

    ax, ay = aim_point(best, offset_y)
    return Det(
        x=ax, y=ay, w=best.w, h=best.h, conf=best.conf,
        cls=best.cls, name=best.name, kind=best.kind,
    )


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


class GainEstimator:
    """Learns how far one mouse count moves the view, from ordinary tracking.

    Over one detection interval the error changes by what the target did minus
    what our own move did::

        d_err = v * dt - k * counts

    That is linear in the two unknowns, so least squares over a window recovers
    both. The point is that the target's motion enters as a fitted term rather
    than as noise to be rejected, which is what made a direct measurement of
    ``k`` fail before: there is no need to separate the two effects when you can
    solve for them together.

    ``k`` is a property of the mouse and the game's sensitivity, so it is
    constant and worth a long, slow, accurate window. ``v`` changes every few
    frames, so once ``k`` is known it is read straight off each sample instead:
    fitting it over the same long window lags a target that changes direction,
    which showed up as large errors against a weaving target.

    Input latency means the counts that moved the view during an interval were
    commanded several intervals earlier. Rather than assume a figure, each
    plausible lag is fitted and the one that best explains the data wins.
    """

    WINDOW = 128
    # Has to span real input latency, not just the display chain: a game with a
    # render queue sits 40-60ms behind. If the search cannot reach the true lag
    # it reports the largest value it has, the gain is then set for a shorter
    # delay than reality, and the loop diverges. Measured: at 60ms latency a
    # range of 9 gave 333px of error, a range of 14 gave 4px.
    LAGS = tuple(range(0, 27))
    K_MIN, K_MAX = 0.15, 40.0
    MIN_SAMPLES = 32
    # Ignore a fit that explains little of the variance, or one taken while the
    # commanded counts barely varied: without spread, move and target motion
    # are indistinguishable.
    MIN_R2 = 0.55
    MIN_SPREAD = 1.5
    # Fits missed in a row before we start accepting flatter command histories,
    # and how many more before the requirement bottoms out.
    STARVE_GRACE = 8
    STARVE_SPAN = 40.0
    SPREAD_FLOOR = 0.6
    RELAXED_R2 = 0.9
    # The fit walks every lag hypothesis, so refit less often now the range is
    # wider. The sensitivity is constant; there is nothing to gain from
    # recomputing it every few frames.
    REFIT_EVERY = 16
    # Fast enough to follow a strafe and direction changes.
    V_ALPHA = 0.45
    K_ALPHA = 0.1

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._cmds: deque[int] = deque(maxlen=64)
        self._samples = {lag: deque(maxlen=self.WINDOW) for lag in self.LAGS}
        self.k: float | None = None
        self.v = 0.0
        self.lag = 0
        self.r2 = 0.0
        self.ready = False
        self._n = 0
        self._starved = 0

    @property
    def starving(self) -> bool:
        """True when the fit has gone unmade long enough to need help."""
        return self._starved > self.STARVE_GRACE

    def add(self, d_err: float, dt: float, counts: int) -> None:
        self._cmds.append(counts)
        for lag in self.LAGS:
            idx = len(self._cmds) - 1 - lag
            if idx >= 0:
                self._samples[lag].append((d_err, dt, self._cmds[idx]))
        self._n += 1

        if self._n % self.REFIT_EVERY == 0 or self.k is None:
            self._fit_k()

        if self.k is not None and dt > 1e-6:
            idx = len(self._cmds) - 1 - self.lag
            c = self._cmds[idx] if idx >= 0 else 0
            v_obs = (d_err + self.k * c) / dt
            if self.ready:
                self.v += (v_obs - self.v) * self.V_ALPHA
            else:
                self.v = v_obs
                self.ready = True

    def _fit_k(self) -> None:
        # Separating our own moves from the target's drift needs the commanded
        # counts to vary, so a flat run of commands is normally skipped. But a
        # loop running on a wrong gain saturates its commands, which flattens
        # them, which blocks the refit that would correct the gain -- a trap it
        # never escapes. So the longer a fit goes unmade, the less variation we
        # insist on. The r-squared check below still throws out bad fits, so
        # the cost of trying is a skipped update rather than a wrong one.
        # The caller dithers the commands once this trips, which restores the
        # variation properly. Easing the requirement only covers the gap until
        # that takes effect, so it stops well short of the flat histories that
        # fit a badly wrong sensitivity.
        need_spread = self.MIN_SPREAD
        need_r2 = self.MIN_R2
        if self._starved > self.STARVE_GRACE:
            slack = (self._starved - self.STARVE_GRACE) / self.STARVE_SPAN
            need_spread = self.MIN_SPREAD * max(self.SPREAD_FLOOR, 1.0 - slack)
            # Thinner evidence has to be better evidence to be believed.
            need_r2 = self.RELAXED_R2

        best: tuple[float, float, int] | None = None
        for lag, buf in self._samples.items():
            if len(buf) < self.MIN_SAMPLES:
                continue
            counts = [c for _, _, c in buf]
            mean_c = sum(counts) / len(counts)
            spread = math.sqrt(sum((c - mean_c) ** 2 for c in counts) / len(counts))
            if spread < need_spread:
                continue

            s11 = s12 = s22 = b1 = b2 = 0.0
            for d_err, dt, c in buf:
                a1, a2 = dt, -float(c)
                s11 += a1 * a1
                s12 += a1 * a2
                s22 += a2 * a2
                b1 += a1 * d_err
                b2 += a2 * d_err
            det = s11 * s22 - s12 * s12
            if abs(det) < 1e-12:
                continue
            v = (b1 * s22 - b2 * s12) / det
            k = (b2 * s11 - b1 * s12) / det
            if not (self.K_MIN <= k <= self.K_MAX):
                continue

            ss_res = sum((de - (v * dt - k * c)) ** 2 for de, dt, c in buf)
            ss_tot = sum(de * de for de, _, _ in buf) or 1e-9
            r2 = 1.0 - ss_res / ss_tot
            if best is None or r2 > best[0]:
                best = (r2, k, lag)

        if best is None:
            self._starved += 1
            return
        r2, k, lag = best
        self.r2 = r2
        if r2 < need_r2:
            self._starved += 1
            return
        self._starved = 0
        self.lag = lag
        self.k = k if self.k is None else self.k + (k - self.k) * self.K_ALPHA


class MoveSolver:
    """Turns a pixel error into mouse counts.

    Shared by the model and colour pipelines so both behave identically.

    Loop gain depends on your in-game sensitivity, which we do not know: one
    mouse count may be one screen pixel or twelve. Measuring that directly turned
    out to be unreliable (surviving input latency needs sums over a window, while
    rejecting the target's own movement needs per-sample pairing, and those
    conflict). So instead of estimating sensitivity we watch whether the error is
    actually shrinking and back the gain off when it is not, which converges from
    either direction.
    """

    # Seconds to close ~63% of the error at speed 1.0.
    BASE_TAU = 0.30
    # Minimum gap between corrections. 1ms for instant response.
    MIN_DT = 0.001
    DEADZONE = 2.5
    # Gain trim bounds and adjustment rates.
    TRIM_MIN = 0.04
    TRIM_BACKOFF = 0.55
    TRIM_RECOVER = 1.03
    # Stick output crosses the target on almost every correction, because the
    # view keeps rotating after the error is satisfied. Those crossings are
    # normal rather than a sign the gain is too hot, so back off gently and
    # keep a usable floor; the mouse schedule collapses to nothing in about
    # six crossings and never recovers.
    VEL_TRIM_MIN = 0.35
    VEL_TRIM_BACKOFF = 0.88
    VEL_TRIM_RECOVER = 1.06
    # Integral term. Correcting purely on present error leaves a standing gap
    # behind anything that moves, because it takes error to generate the speed
    # that keeps up. Accumulating the error supplies that speed, so we ride with
    # the target rather than behind it. Leaks away when the target stops, and is
    # capped so it cannot wind up during a long blocked shot.
    KI = 6.0
    INT_LEAK = 0.97
    INT_CAP = 30.0
    # Slightly under unity: undershooting the target's velocity leaves a small
    # error the proportional term cleans up, while overshooting it drives the
    # view past them, which is the failure that feels worst.
    # Near unity so we match their speed. The P term cleans up any small gap.
    FF_GAIN = 0.96

    # Count-space PID, used once the sensitivity is known.
    #
    # Ki and Kd are deliberately zero. Both were measured as actively harmful
    # here, and the reason is structural rather than a matter of tuning: the
    # derivative of the error is the target's velocity minus our own, and the
    # feedforward already measures the target's velocity directly over a long
    # window, so a D term only re-derives a known quantity from a two-sample
    # difference full of box jitter. An I term exists to cancel a persistent
    # disturbance, which for aiming is the target's motion, and the feedforward
    # cancels that at the source, leaving the integral nothing to accumulate
    # but noise. Measured cost of switching them on: Ki 1.0 was 10% worse,
    # Kd 0.0015 was 12% worse, and Ki above 5 diverged. They are kept here as
    # working terms so the behaviour can be re-checked if the loop changes.
    # Proportional gain is derived from the measured latency rather than fixed.
    # Roughly lag+1 corrections are in flight at once, so commanding more than
    # about 1/(lag+1) of the error per correction means the moves already sent
    # overshoot before the next measurement lands. A fixed 0.22 was stable at
    # the 10ms latency it was tuned at and diverged badly beyond 40ms: 438px of
    # error at 40ms, 1553px at 130ms. Deriving it holds 3-8px across that whole
    # range. The numerator keeps a safety factor below the theoretical 1.0, and
    # at the two-interval latency this was originally tuned at it reproduces
    # the 0.22 the grid search found independently.
    KP_NUM = 0.78
    KP_MIN, KP_MAX = 0.06, 0.48
    # Integral cleans up steady-state error from targets that moved while we
    # were catching up. Small because feedforward handles most of the velocity.
    KI_C = 0.8
    # Derivative damps overshoot. Filtered heavily to not chase box jitter.
    KD_C = 0.002
    CINT_CAP = 4.0
    D_ALPHA = 0.25
    # Backstop for anything the model does not capture, such as a sensitivity
    # estimate that is off or latency that moves around. Crossing the target
    # means the loop is hotter than this setup tolerates, whatever the cause.
    # Low enough to rescue a loop running on a badly wrong gain. At 0.15 the
    # backstop could not pull an over-hot gain back below stable even at full
    # backoff, so it fought the symptom without ever reaching it.
    PID_TRIM_MIN = 0.05
    PID_TRIM_BACKOFF = 0.6
    PID_TRIM_RECOVER = 1.01

    # Watchdog. Everything above assumes the learned sensitivity and latency
    # describe reality. When they do not, the loop can get stuck: too much gain
    # drives commands into their cap, capped commands stop varying, and the
    # estimator needs variation to refit -- so the very thing that would fix
    # the gain is blocked by the bad gain. That deadlock does not clear on its
    # own, which is why it took a restart. Flying past the target repeatedly is
    # proof the model is wrong whatever the cause, so throw it away and measure
    # again from scratch.
    PANIC_ERR = 25.0
    PANIC_HITS = 6.0
    PANIC_COOL = 0.02
    # Finer than the pixel-space deadzone: with the sensitivity known, a
    # correction this small is worth making instead of guessing.
    PID_DEADZONE = 1.5
    # Shared radial deadzone so X and Y cannot keep ticking ±1 out of phase
    # once the crosshair is already on the target. That out-of-phase tick is
    # the circle.
    HOLD_PX = 2.0
    MEAS_ALPHA = 0.65

    def __init__(self, clock=time.perf_counter) -> None:
        self._clock = clock
        self.speed = 2.0
        # A gamepad stick commands a turn rate, not a distance, so "pixels per
        # count" has no fixed value to converge on. Set for stick output so the
        # solver stays on the adaptive pixel-space path instead of fitting a
        # meaningless gain and then dividing by it.
        self.velocity_output = False
        # Tunable PID parameters (can be set via configure_pid)
        self.kp_num = self.KP_NUM
        self.ki_c = self.KI_C
        self.kd_c = self.KD_C
        self.ff_gain = self.FF_GAIN
        self.meas_alpha = self.MEAS_ALPHA
        self._est_x = GainEstimator()
        self._est_y = GainEstimator()
        # Survives reset() so it reads as a running total for the session.
        self.relearns = 0
        self.reset()

    @property
    def sens(self) -> float | None:
        """Learned screen pixels per mouse count, or None until it converges."""
        return self._est_x.k or self._est_y.k

    @property
    def lat_ms(self) -> float:
        """Measured delay between a move and seeing it, in milliseconds."""
        return max(self._est_x.lag, self._est_y.lag) * self._obs_dt * 1000.0

    def configure(self, speed: float) -> None:
        self.speed = _clamp(float(speed), 0.5, 5.0)

    def configure_pid(
        self,
        kp: float | None = None,
        ki: float | None = None,
        kd: float | None = None,
        ff: float | None = None,
        smoothing: float | None = None,
    ) -> None:
        """Update PID parameters live."""
        if kp is not None:
            self.kp_num = _clamp(float(kp), 0.05, 1.5)
        if ki is not None:
            self.ki_c = _clamp(float(ki), 0.0, 5.0)
        if kd is not None:
            self.kd_c = _clamp(float(kd), 0.0, 0.05)
        if ff is not None:
            self.ff_gain = _clamp(float(ff), 0.0, 1.2)
        if smoothing is not None:
            self.meas_alpha = _clamp(float(smoothing), 0.1, 0.95)

    # Every remaining knob the solver reads, with the bounds it stays sane in.
    # The class attributes above are the defaults; assigning an instance
    # attribute of the same name shadows one, so the code that reads it is
    # untouched and the movement behaviour at default values is identical.
    TUNABLES: dict[str, tuple[float, float]] = {
        "BASE_TAU": (0.05, 1.0),
        "MIN_DT": (0.0, 0.02),
        "DEADZONE": (0.0, 10.0),
        "HOLD_PX": (0.0, 10.0),
        "TRIM_MIN": (0.01, 1.0),
        "TRIM_BACKOFF": (0.10, 1.0),
        "TRIM_RECOVER": (1.0, 1.20),
        "VEL_TRIM_MIN": (0.05, 1.0),
        "VEL_TRIM_BACKOFF": (0.10, 1.0),
        "VEL_TRIM_RECOVER": (1.0, 1.20),
        "KI": (0.0, 30.0),
        "INT_LEAK": (0.50, 1.0),
        "INT_CAP": (0.0, 100.0),
        "CINT_CAP": (0.0, 20.0),
        "D_ALPHA": (0.02, 1.0),
        "KP_MIN": (0.01, 0.50),
        "KP_MAX": (0.05, 1.0),
        "PID_TRIM_MIN": (0.01, 1.0),
        "PID_TRIM_BACKOFF": (0.10, 1.0),
        "PID_TRIM_RECOVER": (1.0, 1.20),
        "PID_DEADZONE": (0.0, 8.0),
        "PANIC_ERR": (5.0, 100.0),
        "PANIC_HITS": (1.0, 30.0),
        "PANIC_COOL": (0.0, 1.0),
    }

    def configure_tuning(self, values: dict[str, float] | None) -> None:
        """Override solver constants live. Unknown keys are ignored."""
        for name, raw in (values or {}).items():
            bounds = self.TUNABLES.get(name)
            if bounds is None:
                continue
            try:
                setattr(self, name, _clamp(float(raw), *bounds))
            except (TypeError, ValueError):
                continue
        if self.KP_MAX < self.KP_MIN:
            self.KP_MAX = self.KP_MIN

    @classmethod
    def tuning_defaults(cls) -> dict[str, float]:
        return {name: float(getattr(cls, name)) for name in cls.TUNABLES}

    def reset(self) -> None:
        self._acc_x = 0.0
        self._acc_y = 0.0
        self._last_solve = 0.0
        self._prev_ex = 0.0
        self._prev_ey = 0.0
        self._int_x = 0.0
        self._int_y = 0.0
        self.trim = 1.0
        self._last_obs = 0.0
        self._obs_dt = 0.0
        self._prev_raw_x = 0.0
        self._prev_raw_y = 0.0
        self._have_prev = False
        self._sent_x = 0
        self._sent_y = 0
        self._ff_x = 0.0
        self._ff_y = 0.0
        # Per-axis count-space PID state, indexed [x, y].
        self._filt: list[float | None] = [None, None]
        self._cint = [0.0, 0.0]
        self._cd = [0.0, 0.0]
        self._cprev: list[float | None] = [None, None]
        self._pacc = [0.0, 0.0]
        self._ptrim = [1.0, 1.0]
        self._pprev = [0.0, 0.0]
        self._panic = 0.0
        self._wd_prev = 0.0
        # The learned sensitivity survives: it describes the mouse and the game,
        # not this engagement. The velocity does not, so drop it and let the
        # next few detections re-measure whatever the new target is doing.
        self._est_x.v = self._est_y.v = 0.0

    def relearn(self) -> None:
        """Throw away everything measured and start again from the bootstrap.

        Used when the loop's own behaviour contradicts what it thinks it knows.
        """
        self._est_x.reset()
        self._est_y.reset()
        self._ptrim = [1.0, 1.0]
        self._pacc = [0.0, 0.0]
        self._filt = [None, None]
        self._panic = 0.0
        self.trim = 1.0
        self.relearns += 1

    def _watchdog(self, ex: float, ey: float) -> None:
        """Spot a loop that is flying past the target and force a re-measure.

        Judged on crossing the target by a wide margin, rather than on error
        being large: a target that is simply far away is not a fault, but
        repeatedly landing well past it on alternating sides is.
        """
        err = max(abs(ex), abs(ey))
        prev = self._wd_prev
        signed = ex if abs(ex) >= abs(ey) else ey
        if (
            abs(prev) > self.PANIC_ERR
            and err > self.PANIC_ERR
            and (signed < 0.0) != (prev < 0.0)
        ):
            self._panic += 1.0
        elif err < self.PANIC_ERR:
            self._panic = max(0.0, self._panic - self.PANIC_COOL)
        self._wd_prev = signed

        if self._panic >= self.PANIC_HITS:
            self.relearn()

    def note_move(self, dx: int, dy: int) -> None:
        """Kept for callers; nothing to record now that we do not calibrate."""

    def observe(self, x: float, y: float) -> None:
        """Kept for callers; the solver works purely off the current error."""

    def solve(self, ex: float, ey: float) -> tuple[int, int]:
        now = self._clock()

        # Feed the estimators one sample per detection, pairing this interval's
        # change in error with the counts actually sent last time. This happens
        # before any early return so the fit keeps a continuous record even
        # while sitting inside the deadzone.
        obs_dt = now - self._last_obs if self._last_obs else 0.0
        self._last_obs = now
        if obs_dt > 1e-6:
            self._obs_dt = obs_dt if not self._obs_dt else (
                self._obs_dt + (obs_dt - self._obs_dt) * 0.05
            )
        if self._have_prev and obs_dt > 1e-6 and not self.velocity_output:
            self._est_x.add(ex - self._prev_raw_x, obs_dt, self._sent_x)
            self._est_y.add(ey - self._prev_raw_y, obs_dt, self._sent_y)
        self._prev_raw_x, self._prev_raw_y = ex, ey
        self._have_prev = True
        self._sent_x = self._sent_y = 0

        cap = int(4.0 * (1.0 + self.speed))
        if self.velocity_output:
            kx = ky = None
        else:
            kx = self._est_x.k or self._est_y.k
            ky = self._est_y.k or self._est_x.k
        if kx and ky:
            self._watchdog(ex, ey)
            kx = self._est_x.k or self._est_y.k
            ky = self._est_y.k or self._est_x.k
        if kx and ky:
            fx = self._ff_axis(self._est_x, 0, kx, obs_dt)
            fy = self._ff_axis(self._est_y, 1, ky, obs_dt)
            # Hold still on the proportional term once we are on them so X and
            # Y cannot tick ±1 out of phase. Feedforward still runs: a moving
            # target would otherwise walk out of the hold and get yanked back.
            if math.hypot(ex, ey) < self.HOLD_PX:
                px = py = 0
            elif self._last_solve and (now - self._last_solve) < self.MIN_DT:
                px = py = 0
            else:
                px = self._pid_axis(ex, obs_dt, self._est_x, 0, kx)
                py = self._pid_axis(ey, obs_dt, self._est_y, 1, ky)
                self._last_solve = now
            out_x, out_y = px + fx, py + fy
        else:
            out_x, out_y = self._correct(ex, ey)

        out_x = int(_clamp(out_x, -cap, cap))
        out_y = int(_clamp(out_y, -cap, cap))
        self._sent_x, self._sent_y = out_x, out_y
        return out_x, out_y

    def _ff_axis(self, est: GainEstimator, i: int, k: float, dt: float) -> int:
        if dt <= 1e-6 or not est.ready or k <= 0.0:
            return 0
        cap = 4.0 * (1.0 + self.speed)
        carry = getattr(self, "_ff_x" if i == 0 else "_ff_y")
        carry = _clamp(carry + self.ff_gain * est.v * dt / k, -cap, cap)
        n = int(carry)
        setattr(self, "_ff_x" if i == 0 else "_ff_y", carry - n)
        return n

    def _pid_axis(
        self, err_px: float, dt: float, est: GainEstimator, i: int, k: float
    ) -> int:
        """PID plus feedforward, in mouse counts.

        Dividing the error by the learned sensitivity makes the gains
        dimensionless: Kp of 1 would cancel the whole error in a single move.
        With input latency leaving roughly two corrections in flight, stability
        needs Kp near a third of that, and 0.22 measured best.
        """
        if dt <= 1e-6 or k <= 0.0:
            return 0
        cap = 4.0 * (1.0 + self.speed)

        # Smooth the measurement, not the command. Detection boxes jitter by a
        # couple of pixels, and chasing that jitter is most of what looks like
        # instability. Smoothing position normally costs lag, but here the
        # feedforward carries the target's motion, so the lag is nearly free.
        f = self._filt[i]
        f = err_px if f is None else f + (err_px - f) * self.meas_alpha
        self._filt[i] = f

        if (f < 0.0) != (self._pprev[i] < 0.0) and abs(self._pprev[i]) > 4.0:
            self._ptrim[i] = max(
                self.PID_TRIM_MIN, self._ptrim[i] * self.PID_TRIM_BACKOFF
            )
        else:
            self._ptrim[i] = min(1.0, self._ptrim[i] * self.PID_TRIM_RECOVER)
        self._pprev[i] = f

        u = 0.0
        if abs(f) >= self.PID_DEADZONE:
            e = f / k
            self._cint[i] = _clamp(
                self._cint[i] * self.INT_LEAK + e * dt, -self.CINT_CAP, self.CINT_CAP
            )
            prev = self._cprev[i]
            d_raw = 0.0 if prev is None else (e - prev) / dt
            self._cprev[i] = e
            self._cd[i] += (d_raw - self._cd[i]) * self.D_ALPHA
            kp = _clamp(
                self.kp_num / (est.lag + 1.0), self.KP_MIN, self.KP_MAX
            ) * self._ptrim[i]
            u += kp * e + self.ki_c * self._cint[i] + self.kd_c * self._cd[i]
        else:
            self._cint[i] *= self.INT_LEAK
            self._cprev[i] = f / k

        acc = _clamp(self._pacc[i] + u, -cap, cap)
        n = int(acc)
        self._pacc[i] = acc - n
        return n

    def _feedforward(self, dt: float) -> tuple[int, int]:
        """Counts that match the target's measured screen velocity."""
        if dt <= 1e-6:
            return 0, 0
        cap = 4.0 * (1.0 + self.speed)
        out = []
        for est, attr in ((self._est_x, "_ff_x"), (self._est_y, "_ff_y")):
            if not est.ready or not est.k:
                out.append(0)
                continue
            carry = _clamp(
                getattr(self, attr) + self.ff_gain * est.v * dt / est.k, -cap, cap
            )
            n = int(carry)
            setattr(self, attr, carry - n)
            out.append(n)
        return out[0], out[1]

    def _correct(self, ex: float, ey: float) -> tuple[int, int]:
        if math.hypot(ex, ey) < self.DEADZONE:
            self._acc_x = self._acc_y = 0.0
            self._prev_ex = self._prev_ey = 0.0
            self._int_x = self._int_y = 0.0
            return 0, 0

        # Pace corrections. Detection runs far faster than the game can show the
        # result of a move, so firing on every detection stacks a dozen
        # corrections for the same error in flight at once and the view runs
        # away. Waiting keeps roughly one correction outstanding, which is what
        # keeps the loop stable without knowing the sensitivity.
        now = self._clock()
        if self._last_solve and (now - self._last_solve) < self.MIN_DT:
            return 0, 0
        elapsed = now - self._last_solve if self._last_solve else self.MIN_DT
        self._last_solve = now

        # An error that flips sign means we crossed the target: the gain is too
        # hot for this sensitivity, so back off hard and creep back up. Judging
        # by "did the error grow" instead would also fire whenever the target
        # simply ran away from us, throttling the gain to nothing while chasing.
        flipped = (
            (ex < 0.0) != (self._prev_ex < 0.0) and abs(self._prev_ex) > self.DEADZONE
        ) or (
            (ey < 0.0) != (self._prev_ey < 0.0) and abs(self._prev_ey) > self.DEADZONE
        )
        if self.velocity_output:
            t_min, t_back, t_rec = (
                self.VEL_TRIM_MIN, self.VEL_TRIM_BACKOFF, self.VEL_TRIM_RECOVER
            )
        else:
            t_min, t_back, t_rec = (
                self.TRIM_MIN, self.TRIM_BACKOFF, self.TRIM_RECOVER
            )
        if flipped:
            self.trim = max(t_min, self.trim * t_back)
            # Crossing the target means the accumulated speed is now wrong.
            self._int_x = self._int_y = 0.0
        else:
            self.trim = min(1.0, self.trim * t_rec)
        self._prev_ex, self._prev_ey = ex, ey

        self._int_x = _clamp(
            self._int_x * self.INT_LEAK + ex * elapsed, -self.INT_CAP, self.INT_CAP
        )
        self._int_y = _clamp(
            self._int_y * self.INT_LEAK + ey * elapsed, -self.INT_CAP, self.INT_CAP
        )
        ex += self.KI * self._int_x
        ey += self.KI * self._int_y

        # Time-based gain so the feel does not change with detection Hz.
        gain = 1.0 - math.exp(-_clamp(elapsed, 1e-4, 0.05) / (self.BASE_TAU / self.speed))
        gain *= self.trim

        self._acc_x += ex * gain
        self._acc_y += ey * gain

        # Cap per-move counts so a target snapping in cannot yank the view, and
        # so one bad step cannot throw the view far on a high-sensitivity setup.
        cap = 3.0 + self.speed * 3.0
        self._acc_x = _clamp(self._acc_x, -cap, cap)
        self._acc_y = _clamp(self._acc_y, -cap, cap)

        # Carry the remainder rather than rounding a sub-count correction up to a
        # whole count. Rounding up fires every detection, and because X and Y
        # round independently the result is a circle around the target.
        ix = int(self._acc_x)
        iy = int(self._acc_y)
        self._acc_x -= ix
        self._acc_y -= iy
        return ix, iy


class Aimer:
    MISS_LIMIT = 6

    def __init__(self, clock=time.perf_counter) -> None:
        self.solver = MoveSolver(clock)
        self.last_dx = 0
        self.last_dy = 0
        self.err_x = 0.0
        self.err_y = 0.0
        self._lock: Det | None = None
        self._misses = 0
        self._fresh = False
        
        # Advanced aim settings
        self._max_speed = 0.0  # 0 = unlimited
        self._acceleration = 0.0  # 0 = instant, 1 = slow ramp
        self._jitter = 0.0  # humanization jitter in pixels
        self._flick_reduction = 0.0  # 0 = off, 1 = max smoothing
        self._easing = "linear"
        self._current_speed = 0.0  # for acceleration
        self._last_move_time = 0.0
        self._smoothed_dx = 0.0
        self._smoothed_dy = 0.0
        self._post_acc_x = 0.0
        self._post_acc_y = 0.0

    @property
    def speed(self) -> float:
        return self.solver.speed

    def configure(self, track_speed: float) -> None:
        self.solver.configure(track_speed)
    
    def configure_advanced(
        self,
        max_speed: float | None = None,
        acceleration: float | None = None,
        jitter: float | None = None,
        flick_reduction: float | None = None,
        easing: str | None = None,
    ) -> None:
        """Configure advanced aim settings."""
        if max_speed is not None:
            self._max_speed = max(0.0, float(max_speed))
        if acceleration is not None:
            self._acceleration = _clamp(float(acceleration), 0.0, 1.0)
        if jitter is not None:
            self._jitter = max(0.0, float(jitter))
        if flick_reduction is not None:
            self._flick_reduction = _clamp(float(flick_reduction), 0.0, 1.0)
        if easing is not None and easing in ("linear", "ease_in", "ease_out", "ease_in_out"):
            self._easing = easing

    def configure_pid(
        self,
        kp: float | None = None,
        ki: float | None = None,
        kd: float | None = None,
        ff: float | None = None,
        smoothing: float | None = None,
    ) -> None:
        """Update PID parameters live."""
        self.solver.configure_pid(kp=kp, ki=ki, kd=kd, ff=ff, smoothing=smoothing)

    def configure_tuning(self, values: dict[str, float] | None) -> None:
        """Override the solver's internal constants live."""
        self.solver.configure_tuning(values)

    @staticmethod
    def solver_defaults() -> dict[str, float]:
        return MoveSolver.tuning_defaults()

    def set_velocity_output(self, enabled: bool) -> None:
        """Tell the solver whether output drives a stick rather than a mouse."""
        was = self.solver.velocity_output
        self.solver.velocity_output = bool(enabled)
        if was != self.solver.velocity_output:
            # The two paths keep different state; carrying it across would feed
            # the new one stale integrals and a stale gain fit.
            self.solver.reset()

    def set_aspect(self, screen_w: float, screen_h: float) -> None:
        pass

    def reset(self) -> None:
        self._lock = None
        self._misses = 0
        self._fresh = False
        self.last_dx = self.last_dy = 0
        self.err_x = self.err_y = 0.0
        self._post_acc_x = self._post_acc_y = 0.0
        self.solver.reset()

    def observe(self, target: Det | None, dets: list[Det], *, offset_y: float = 0.5) -> None:
        if target is not None:
            self._lock = target
            self._misses = 0
            self._fresh = True
            self.solver.observe(target.x, target.y)
        else:
            self._misses += 1
            if self._misses > self.MISS_LIMIT:
                self._lock = None
                self._fresh = False

    @property
    def target(self) -> Det | None:
        return self._lock

    def note_move(self, dx: int, dy: int) -> None:
        self.solver.note_move(dx, dy)

    def step(self, cx: float, cy: float, dt: float = 0.002) -> tuple[int, int]:
        self.last_dx = self.last_dy = 0

        # One move per detection. Moving again before re-measuring just piles
        # more correction onto an error we have already acted on.
        if self._lock is None or not self._fresh:
            self._current_speed *= 0.8  # decay acceleration when idle
            return 0, 0
        self._fresh = False

        ex = self._lock.x - cx
        ey = self._lock.y - cy
        self.err_x, self.err_y = ex, ey

        ix, iy = self.solver.solve(ex, ey)
        
        # Apply advanced post-processing
        ix, iy = self._apply_advanced(ix, iy, ex, ey, dt)
        
        self.last_dx, self.last_dy = ix, iy
        return ix, iy
    
    def _apply_advanced(self, dx: int, dy: int, ex: float, ey: float, dt: float) -> tuple[int, int]:
        """Apply advanced aim features: speed limit, acceleration, easing, flick reduction, humanization."""
        if dx == 0 and dy == 0:
            return 0, 0
        
        fx, fy = float(dx), float(dy)
        mag = math.hypot(fx, fy)
        
        if mag < 0.01:
            return 0, 0
        
        # 1. Apply easing curve based on distance to target
        err_mag = math.hypot(ex, ey)
        if self._easing != "linear" and err_mag > 0:
            # Normalize error to 0-1 range (assuming max useful range ~200px)
            t = _clamp(err_mag / 200.0, 0.0, 1.0)
            ease_factor = self._ease(t)
            # Scale movement by easing factor
            fx *= ease_factor
            fy *= ease_factor
            mag = math.hypot(fx, fy)
        
        # 2. Apply flick reduction (smooth large movements)
        if self._flick_reduction > 0 and mag > 3:
            alpha = 1.0 - self._flick_reduction * 0.7
            fx = self._smoothed_dx + (fx - self._smoothed_dx) * alpha
            fy = self._smoothed_dy + (fy - self._smoothed_dy) * alpha
            self._smoothed_dx = fx
            self._smoothed_dy = fy
            mag = math.hypot(fx, fy)
        else:
            self._smoothed_dx = fx
            self._smoothed_dy = fy
        
        # 3. Apply acceleration (ramp up speed)
        if self._acceleration > 0:
            # Target speed based on how far we need to go
            target_speed = mag
            # Ramp rate: lower acceleration = faster ramp
            ramp = 1.0 - self._acceleration * 0.95
            self._current_speed += (target_speed - self._current_speed) * ramp
            
            if mag > 0.01:
                scale = self._current_speed / mag
                fx *= scale
                fy *= scale
                mag = self._current_speed
        
        # 4. Apply max speed limit
        if self._max_speed > 0 and mag > self._max_speed:
            scale = self._max_speed / mag
            fx *= scale
            fy *= scale
        
        # 5. Apply humanization jitter
        if self._jitter > 0:
            import random
            fx += random.uniform(-self._jitter, self._jitter)
            fy += random.uniform(-self._jitter, self._jitter)
        
        fx += self._post_acc_x
        fy += self._post_acc_y
        ix, iy = int(round(fx)), int(round(fy))
        self._post_acc_x = fx - ix
        self._post_acc_y = fy - iy
        return ix, iy
    
    def _ease(self, t: float) -> float:
        """Apply easing curve. t is 0-1, returns 0-1."""
        if self._easing == "ease_in":
            # Start slow, end fast (quadratic)
            return t * t
        elif self._easing == "ease_out":
            # Start fast, end slow (inverse quadratic)
            return 1.0 - (1.0 - t) * (1.0 - t)
        elif self._easing == "ease_in_out":
            # Smooth both ends (smoothstep)
            return t * t * (3.0 - 2.0 * t)
        return t  # linear
