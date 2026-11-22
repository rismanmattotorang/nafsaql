/* Nafs-Aql project site: optional enhancements only.
   The pages are fully usable without this file. No network requests, no cookies. */
(function () {
  "use strict";
  var root = document.documentElement;
  root.classList.add("js");

  var KEY = "nafsaql-site-theme";
  function stored() {
    try { return window.localStorage.getItem(KEY); } catch (e) { return null; }
  }
  function store(value) {
    try { window.localStorage.setItem(KEY, value); } catch (e) { /* private window: keep it for this page only */ }
  }
  var saved = stored();
  if (saved === "light" || saved === "dark") { root.setAttribute("data-theme", saved); }

  function currentTheme() {
    var explicit = root.getAttribute("data-theme");
    if (explicit) { return explicit; }
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function ready() {
    var themeBtn = document.querySelector(".theme-toggle");
    function label() {
      if (!themeBtn) { return; }
      var next = currentTheme() === "dark" ? "light" : "dark";
      themeBtn.setAttribute("aria-label", "Switch to the " + next + " theme");
      themeBtn.setAttribute("title", "Switch to the " + next + " theme");
    }
    if (themeBtn) {
      label();
      themeBtn.addEventListener("click", function () {
        var next = currentTheme() === "dark" ? "light" : "dark";
        root.setAttribute("data-theme", next);
        store(next);
        label();
      });
    }

    var navBtn = document.querySelector(".nav-toggle");
    var nav = document.getElementById("site-nav");
    if (navBtn && nav) {
      navBtn.addEventListener("click", function () {
        var open = nav.classList.toggle("open");
        navBtn.setAttribute("aria-expanded", open ? "true" : "false");
      });
      document.addEventListener("keydown", function (ev) {
        if (ev.key === "Escape" && nav.classList.contains("open")) {
          nav.classList.remove("open");
          navBtn.setAttribute("aria-expanded", "false");
          navBtn.focus();
        }
      });
    }

    var blocks = document.querySelectorAll("[data-copy]");
    Array.prototype.forEach.call(blocks, function (block) {
      var bar = block.querySelector(".terminal-bar");
      var pre = block.querySelector("pre");
      if (!bar || !pre || !navigator.clipboard) { return; }
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "copy-btn";
      btn.textContent = "Copy";
      btn.setAttribute("aria-label", "Copy the commands");
      btn.addEventListener("click", function () {
        var lines = pre.querySelectorAll("[data-cmd]");
        var text = Array.prototype.map.call(lines, function (l) { return l.textContent; }).join("\n");
        navigator.clipboard.writeText(text || pre.textContent).then(function () {
          btn.textContent = "Copied";
          window.setTimeout(function () { btn.textContent = "Copy"; }, 1600);
        }, function () { /* clipboard refused: leave the text selectable */ });
      });
      bar.appendChild(btn);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", ready);
  } else {
    ready();
  }
})();
