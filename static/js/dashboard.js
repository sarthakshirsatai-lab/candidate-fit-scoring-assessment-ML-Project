document.addEventListener("DOMContentLoaded", function () {
  var PAGE_SIZE = 10;

  var uploadBtn = document.getElementById("upload-btn");
  var fileInput = document.getElementById("csv-input");
  var errorBox = document.getElementById("upload-error");
  var scoringStatus = document.getElementById("scoring-status");
  var paginationEl = document.getElementById("pagination-controls");
  var allRows = Array.prototype.slice.call(document.querySelectorAll(".candidate-row"));

  var currentFilter = "all";
  var currentPage = 1;

  // -------------------------------------------------------------------
  // Upload + loading state
  // -------------------------------------------------------------------

  if (uploadBtn && fileInput) {
    uploadBtn.addEventListener("click", function () {
      fileInput.click();
    });

    fileInput.addEventListener("change", function () {
      if (!fileInput.files || fileInput.files.length === 0) {
        return;
      }
      var formData = new FormData();
      formData.append("file", fileInput.files[0]);

      uploadBtn.disabled = true;
      hideError();
      showScoringStatus();

      fetch("/upload", { method: "POST", body: formData })
        .then(function (response) {
          return response.json().then(function (data) {
            return { ok: response.ok, data: data };
          });
        })
        .then(function (result) {
          if (result.ok) {
            window.location.reload();
          } else {
            showError(result.data.error || "Upload failed.");
            resetUploadButton();
          }
        })
        .catch(function (err) {
          showError("Upload failed: " + err);
          resetUploadButton();
        });
    });
  }

  function resetUploadButton() {
    uploadBtn.disabled = false;
    fileInput.value = "";
    hideScoringStatus();
  }

  function showScoringStatus() {
    if (scoringStatus) scoringStatus.hidden = false;
  }

  function hideScoringStatus() {
    if (scoringStatus) scoringStatus.hidden = true;
  }

  function showError(message) {
    if (!errorBox) return;
    errorBox.textContent = message;
    errorBox.hidden = false;
  }

  function hideError() {
    if (!errorBox) return;
    errorBox.hidden = true;
    errorBox.textContent = "";
  }

  // -------------------------------------------------------------------
  // Breakdown toggle
  // -------------------------------------------------------------------

  document.querySelectorAll('[data-action="toggle-breakdown"]').forEach(function (btn) {
    btn.addEventListener("click", function () {
      var row = btn.closest(".candidate-row");
      var panel = row.querySelector(".candidate-breakdown");
      var isHidden = panel.hasAttribute("hidden");
      if (isHidden) {
        panel.removeAttribute("hidden");
        btn.textContent = "Hide breakdown";
      } else {
        panel.setAttribute("hidden", "");
        btn.textContent = "See breakdown";
      }
    });
  });

  // -------------------------------------------------------------------
  // Shortlist / Pass decisions
  // -------------------------------------------------------------------

  document.querySelectorAll(".btn-decision").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var row = btn.closest(".candidate-row");
      var rowId = row.getAttribute("data-row-id");
      var decision = btn.getAttribute("data-decision");
      var alreadyActive = btn.classList.contains("active");
      var newDecision = alreadyActive ? "Pending" : decision;

      fetch("/decision/" + rowId, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ decision: newDecision }),
      })
        .then(function (response) {
          return response.json();
        })
        .then(function (data) {
          if (data.ok) {
            row.querySelectorAll(".btn-decision").forEach(function (b) {
              b.classList.remove("active");
            });
            if (newDecision !== "Pending") {
              btn.classList.add("active");
            }
          }
        });
    });
  });

  // -------------------------------------------------------------------
  // Stat-card filtering + pagination
  // -------------------------------------------------------------------

  document.querySelectorAll(".stat-card").forEach(function (card) {
    card.addEventListener("click", function () {
      var filter = card.getAttribute("data-filter");
      if (filter === "all") {
        currentFilter = "all";
      } else {
        currentFilter = currentFilter === filter ? "all" : filter;
      }
      currentPage = 1;
      updateActiveCard();
      renderPage();
    });
  });

  function updateActiveCard() {
    document.querySelectorAll(".stat-card").forEach(function (card) {
      var filter = card.getAttribute("data-filter");
      card.classList.toggle("active", currentFilter !== "all" && filter === currentFilter);
    });
  }

  function getFilteredRows() {
    if (currentFilter === "all") return allRows;
    return allRows.filter(function (row) {
      return row.getAttribute("data-tier") === currentFilter;
    });
  }

  function renderPage() {
    var filtered = getFilteredRows();
    var totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
    if (currentPage > totalPages) currentPage = totalPages;
    if (currentPage < 1) currentPage = 1;

    var start = (currentPage - 1) * PAGE_SIZE;
    var end = start + PAGE_SIZE;
    var visible = filtered.slice(start, end);

    allRows.forEach(function (row) {
      row.hidden = visible.indexOf(row) === -1;
    });

    renderPaginationControls(totalPages);
  }

  function renderPaginationControls(totalPages) {
    if (!paginationEl) return;
    paginationEl.innerHTML = "";

    if (totalPages <= 1) {
      paginationEl.hidden = true;
      return;
    }
    paginationEl.hidden = false;

    paginationEl.appendChild(
      makePageButton("Previous", currentPage - 1, currentPage === 1, false)
    );

    getPageNumbers(currentPage, totalPages).forEach(function (item) {
      if (item === "...") {
        var ellipsis = document.createElement("span");
        ellipsis.className = "pagination-ellipsis";
        ellipsis.textContent = "…";
        paginationEl.appendChild(ellipsis);
      } else {
        paginationEl.appendChild(makePageButton(String(item), item, false, item === currentPage));
      }
    });

    paginationEl.appendChild(
      makePageButton("Next", currentPage + 1, currentPage === totalPages, false)
    );
  }

  function makePageButton(label, targetPage, disabled, isActive) {
    var btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = label;
    if (disabled) btn.disabled = true;
    if (isActive) btn.classList.add("active");
    btn.addEventListener("click", function () {
      currentPage = targetPage;
      renderPage();
      var resultsSection = document.getElementById("results-section");
      if (resultsSection) resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    });
    return btn;
  }

  function getPageNumbers(current, total) {
    if (total <= 7) {
      var all = [];
      for (var i = 1; i <= total; i++) all.push(i);
      return all;
    }
    var pages = [1];
    if (current > 4) pages.push("...");
    var start = Math.max(2, current - 1);
    var end = Math.min(total - 1, current + 1);
    for (var p = start; p <= end; p++) pages.push(p);
    if (current < total - 3) pages.push("...");
    pages.push(total);
    return pages;
  }

  if (allRows.length > 0) {
    renderPage();
  }
});
