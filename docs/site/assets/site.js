/* Progressive enhancement only: the page is complete without this file.
 * Copy buttons appear when a clipboard exists; the hero animation pauses
 * off-screen and gains a replay button. */

"use strict";

// Copy-to-clipboard on the terminal blocks.
if (navigator.clipboard) {
  for (const button of document.querySelectorAll(".copy[data-copy]")) {
    button.hidden = false;
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(button.dataset.copy);
        button.textContent = "copied";
        setTimeout(() => { button.textContent = "copy"; }, 1500);
      } catch { /* the command is right there to select */ }
    });
  }
}

const hero = document.getElementById("run");
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

if (hero && !reducedMotion) {
  // Battery: pause the 12s loop while the hero is not on screen.
  if ("IntersectionObserver" in window) {
    new IntersectionObserver((entries) => {
      for (const entry of entries) {
        hero.classList.toggle("offscreen", !entry.isIntersecting);
      }
    }).observe(hero);
  }

  const replay = document.getElementById("replay-hero");
  if (replay) {
    replay.hidden = false;
    replay.addEventListener("click", () => {
      hero.classList.add("reset");
      void hero.getBoundingClientRect(); // flush, so removal restarts the clock
      hero.classList.remove("reset");
    });
  }
}
