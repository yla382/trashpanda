// The sidebar: the trace files of the trace folder, newest first. A click opens one (the page loads again with ?name=...).
// It is read-only: nothing here changes a file.

const KEY = "trash-panda-files-collapsed";

export function openedName() {
  return new URLSearchParams(location.search).get("name");
}

export function traceUrl() {
  const name = openedName();
  return "/trace.json" + (name ? "?name=" + encodeURIComponent(name) : "");
}

function size(n) {
  if (n < 1024) return n + " B";
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
  return (n / 1024 / 1024).toFixed(1) + " MB";
}

function remembered() {
  try { return localStorage.getItem(KEY) === "1"; } catch (e) { return false; }
}

function remember(collapsed) {
  try { localStorage.setItem(KEY, collapsed ? "1" : "0"); } catch (e) { /* the page works without it */ }
}

// The list is hidden when the server has no folder of traces to show (and shown, with a note, when the folder is empty).
export function buildFiles(nav) {
  return fetch("/api/traces")
    .then((r) => r.json())
    .then((listing) => {
      if (!listing.folder) return;
      const name = openedName() || listing.current;
      const inFolder = openedName() ? true : listing.current_in_folder;
      nav.hidden = false;
      const toggle = nav.querySelector("button.collapse");
      const apply = (collapsed) => {
        nav.classList.toggle("collapsed", collapsed);
        toggle.setAttribute("aria-expanded", String(!collapsed));
        toggle.textContent = collapsed ? "›" : "‹";
        toggle.setAttribute("aria-label", collapsed ? "Show the recorded runs" : "Hide the recorded runs");
      };
      apply(remembered());
      toggle.addEventListener("click", () => {
        const collapsed = !nav.classList.contains("collapsed");
        apply(collapsed);
        remember(collapsed);
      });

      const folder = nav.querySelector(".folder");
      folder.textContent = listing.folder;
      folder.title = listing.folder;                                 // a long path is cut short in the box; the whole of it is the tooltip
      const list = nav.querySelector("ul");
      if (!listing.files.length) {
        const li = document.createElement("li");
        li.className = "none";
        li.textContent = "No trace files in this folder yet. Record one with TRASH_PANDA_TRACE=1.";
        list.appendChild(li);
      }
      listing.files.forEach((f) => {
        const li = document.createElement("li");
        const a = document.createElement("a");
        a.href = "?name=" + encodeURIComponent(f.name);
        if (inFolder && f.name === name) a.setAttribute("aria-current", "page");
        const title = document.createElement("span");
        title.className = "fname";
        title.textContent = f.name.replace(/\.json$/, "");
        const sub = document.createElement("span");
        sub.className = "fsub";
        sub.textContent = new Date(f.modified * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }) + " · " + size(f.size);
        a.append(title, sub);
        li.appendChild(a);
        list.appendChild(li);
      });
      if (!inFolder) {                                             // a file from outside the folder was opened
        const note = nav.querySelector(".outside");
        note.textContent = "Showing " + name + ", which is not in this folder.";
        note.hidden = false;
      }
    })
    .catch(() => { /* no list: the page still shows the trace */ });
}
