/* Smart Crop Recommendation - progressive enhancement only.
   Every page works with JavaScript disabled; this file improves the experience.
   No external dependencies, no network requests. */

(function () {
  "use strict";

  /* ---- Mobile navigation ------------------------------------------- */
  var toggle = document.querySelector(".nav-toggle");
  var nav = document.getElementById("primary-nav");

  if (toggle && nav) {
    toggle.addEventListener("click", function () {
      var open = nav.classList.toggle("is-open");
      toggle.setAttribute("aria-expanded", String(open));
      toggle.setAttribute("aria-label", open ? "Close menu" : "Open menu");
    });

    // Close the menu when focus leaves it, so keyboard users are not trapped.
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && nav.classList.contains("is-open")) {
        nav.classList.remove("is-open");
        toggle.setAttribute("aria-expanded", "false");
        toggle.focus();
      }
    });
  }

  /* ---- Broken image fallback ---------------------------------------
     If a crop image 404s after deployment, swap in the shared placeholder
     rather than leaving a broken icon. Guarded so it cannot loop.        */
  var FALLBACK = "_unknown.svg";

  document.querySelectorAll("img[src*='/crops/']").forEach(function (img) {
    img.addEventListener("error", function handleError() {
      img.removeEventListener("error", handleError);
      if (img.src.indexOf(FALLBACK) !== -1) return;
      img.src = img.src.replace(/\/crops\/[^/]+$/, "/crops/" + FALLBACK);
      img.alt = "Crop image not available";
    });
  });

  /* ---- Submit-time feedback ----------------------------------------
     Server-side validation is authoritative; this only stops accidental
     double submits.                                                     */
  document.querySelectorAll("form").forEach(function (form) {
    form.addEventListener("submit", function () {
      var button = form.querySelector("button[type='submit']");
      if (!button || button.disabled) return;
      window.setTimeout(function () {
        button.disabled = true;
        button.textContent = "Working\u2026";
      }, 0);
    });
  });

  /* ---- Move focus to the first invalid field ------------------------ */
  var firstInvalid = document.querySelector("[aria-invalid='true']");
  if (firstInvalid) firstInvalid.focus();

  /* ---- Reduced motion preference ------------------------------------ */
  var reduceMotion = window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---- Drifting leaves in the hero -----------------------------------
     Purely decorative and never blocks anything. Skipped entirely when
     the visitor has asked for reduced motion.                            */
  var hero = document.querySelector(".hero");
  if (hero && !reduceMotion) {
    var field = document.createElement("div");
    field.className = "leaf-field";
    field.setAttribute("aria-hidden", "true");
    hero.insertBefore(field, hero.firstChild);

    var LEAF_PATH =
      "M12 2c5 0 9 4 9 9 0 6-6 11-9 11S3 17 3 11c0-5 4-9 9-9z" +
      "M12 3v18";
    var LEAF_COUNT = window.innerWidth < 760 ? 6 : 11;

    for (var i = 0; i < LEAF_COUNT; i++) {
      var size = 14 + Math.random() * 16;
      var leaf = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      leaf.setAttribute("viewBox", "0 0 24 24");
      leaf.setAttribute("width", String(size));
      leaf.setAttribute("height", String(size));
      leaf.setAttribute("fill", "none");
      leaf.setAttribute("stroke", "currentColor");
      leaf.setAttribute("stroke-width", "1.6");
      leaf.setAttribute("stroke-linecap", "round");
      leaf.classList.add("leaf");

      var path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", LEAF_PATH);
      leaf.appendChild(path);

      var left = Math.random() * 100;
      var duration = 9 + Math.random() * 8;
      var delay = Math.random() * -duration;
      var swayDuration = 3 + Math.random() * 2;

      leaf.style.left = left + "%";
      leaf.style.animationDuration = duration + "s, " + swayDuration + "s";
      leaf.style.animationDelay = delay + "s, " + delay + "s";

      field.appendChild(leaf);
    }
  }

  /* ---- Probability bars fill in after the cards land ----------------- */
  var meters = document.querySelectorAll(".meter-fill");
  if (meters.length) {
    window.requestAnimationFrame(function () {
      window.requestAnimationFrame(function () {
        meters.forEach(function (meter) { meter.classList.add("is-filled"); });
      });
    });
  }

  /* ---- Sound effects ---------------------------------------------------
     Short tones synthesised with the Web Audio API - no audio files to
     fetch, nothing to load, and it works offline. Muted by default only
     if the visitor previously chose to mute it (remembered locally); a
     toggle in the header lets them turn it off at any time. Browsers
     block audio before the first user gesture, so the context is created
     lazily and anything queued before that gesture is played on it.    */
  var SOUND_KEY = "scr-sound-muted";
  var soundToggle = document.querySelector("[data-sound-toggle]");
  var muted = window.localStorage && window.localStorage.getItem(SOUND_KEY) === "1";
  var audioCtx = null;
  var pending = null;
  var unlocked = false;

  function setToggleState() {
    if (!soundToggle) return;
    soundToggle.setAttribute("aria-pressed", String(!muted));
    soundToggle.classList.toggle("is-muted", muted);
    soundToggle.setAttribute(
      "aria-label",
      muted ? "Turn sound effects on" : "Turn sound effects off"
    );
  }
  setToggleState();

  function getContext() {
    if (!audioCtx) {
      var AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass) return null;
      audioCtx = new AudioContextClass();
    }
    return audioCtx;
  }

  function tone(ctx, freq, startAt, dur, gainPeak, type) {
    var osc = ctx.createOscillator();
    var gain = ctx.createGain();
    osc.type = type || "sine";
    osc.frequency.setValueAtTime(freq, startAt);
    gain.gain.setValueAtTime(0, startAt);
    gain.gain.linearRampToValueAtTime(gainPeak, startAt + 0.015);
    gain.gain.exponentialRampToValueAtTime(0.0001, startAt + dur);
    osc.connect(gain).connect(ctx.destination);
    osc.start(startAt);
    osc.stop(startAt + dur + 0.05);
  }

  function playClick() {
    if (muted) return;
    var ctx = getContext();
    if (!ctx || ctx.state === "suspended") { pending = playClick; return; }
    tone(ctx, 720, ctx.currentTime, 0.07, 0.05, "sine");
  }

  function playChime() {
    if (muted) return;
    var ctx = getContext();
    if (!ctx || ctx.state === "suspended") { pending = playChime; return; }
    // A short, gentle rising arpeggio - a "good result" cue, not an alarm.
    var notes = [523.25, 659.25, 783.99, 1046.5]; // C5 E5 G5 C6
    var now = ctx.currentTime;
    notes.forEach(function (freq, index) {
      tone(ctx, freq, now + index * 0.09, 0.5, 0.045, "sine");
    });
  }

  function unlockAudio() {
    if (unlocked) return;
    unlocked = true;
    var ctx = getContext();
    if (ctx && ctx.state === "suspended") {
      ctx.resume().then(function () {
        if (pending) { var fn = pending; pending = null; fn(); }
      });
    } else if (pending) {
      var fn = pending; pending = null; fn();
    }
  }
  ["pointerdown", "keydown", "touchstart"].forEach(function (evt) {
    document.addEventListener(evt, unlockAudio, { once: true, passive: true });
  });

  if (soundToggle) {
    soundToggle.addEventListener("click", function () {
      muted = !muted;
      if (window.localStorage) {
        window.localStorage.setItem(SOUND_KEY, muted ? "1" : "0");
      }
      setToggleState();
      if (!muted) playClick();
    });
  }

  document.querySelectorAll(".btn, .mode-card, .nav-toggle").forEach(function (el) {
    el.addEventListener("click", playClick);
  });

  // A results page that actually produced cards gets the success chime.
  if (document.querySelector(".result-card")) {
    window.setTimeout(playChime, 250);
  }
})();
