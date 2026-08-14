(function () {
  "use strict";

  function deg(n) {
    return (n * Math.PI) / 180;
  }

  function wrap90(t) {
    var x = t + 90;
    x = x - 180 * Math.floor(x / 180);
    return x - 90;
  }

  function drawBox(g, cx, cy, w, h, angle, extra) {
    var rad = deg(angle);
    var hw = w / 2;
    var hh = h / 2;
    var corners = [
      [-hw, -hh],
      [hw, -hh],
      [hw, hh],
      [-hw, hh]
    ];
    var pts = corners.map(function (p) {
      var x = p[0] * Math.cos(rad) - p[1] * Math.sin(rad) + cx;
      var y = p[0] * Math.sin(rad) + p[1] * Math.cos(rad) + cy;
      return x.toFixed(1) + "," + y.toFixed(1);
    });
    var poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    poly.setAttribute("points", pts.join(" "));
    g.appendChild(poly);
    if (extra) {
      var tipX = hw * Math.cos(rad) + cx;
      var tipY = hw * Math.sin(rad) + cy;
      var tick = document.createElementNS("http://www.w3.org/2000/svg", "line");
      tick.setAttribute("x1", String(tipX));
      tick.setAttribute("y1", String(tipY));
      tick.setAttribute("x2", String(tipX + 10 * Math.cos(rad - 0.4)));
      tick.setAttribute("y2", String(tipY + 10 * Math.sin(rad - 0.4)));
      g.appendChild(tick);
    }
  }

  function setupRotator() {
    var svg = document.getElementById("rotator");
    if (!svg) return;
    var slider = document.getElementById("theta-pred");
    var out = document.getElementById("theta-out");
    var square = document.getElementById("square-toggle");
    var gt = document.getElementById("gt-box");
    var pred = document.getElementById("pred-box");
    var coordGap = document.getElementById("coord-gap");
    var gwdScore = document.getElementById("gwd-score");
    var squareState = document.getElementById("square-state");
    var coordMark = document.getElementById("coord-mark");

    function render() {
      var theta = Number(slider.value);
      var isSquare = square.checked;
      out.textContent = theta + "°";
      var w = isSquare ? 70 : 140;
      var h = isSquare ? 70 : 36;
      var gtAngle = 20;
      gt.replaceChildren();
      pred.replaceChildren();
      coordMark.replaceChildren();
      drawBox(gt, 220, 140, w, h, gtAngle, false);
      drawBox(pred, 220, 140, w, h, theta, true);
      var seam = Math.abs(wrap90(theta) - wrap90(gtAngle));
      var naive = Math.abs(theta - gtAngle);
      var gwd = Math.sqrt(Math.pow(0.15 * (w - h) / 100, 2) + Math.pow(seam / 90, 2));
      coordGap.textContent = naive.toFixed(1) + "° raw heading";
      gwdScore.textContent = gwd.toFixed(3) + " (schematic)";
      squareState.textContent = isSquare ? "near-square (heading unconstrained)" : "elongated";
      var jump = document.createElementNS("http://www.w3.org/2000/svg", "line");
      jump.setAttribute("x1", "430");
      jump.setAttribute("y1", String(40 + naive));
      jump.setAttribute("x2", "600");
      jump.setAttribute("y2", String(40 + naive));
      coordMark.appendChild(jump);
    }
    slider.addEventListener("input", render);
    square.addEventListener("change", render);
    render();
  }

  function setupForest() {
    var chips = document.querySelectorAll(".chip[data-filter]");
    if (!chips.length) return;
    chips.forEach(function (chip) {
      chip.addEventListener("click", function () {
        chips.forEach(function (c) { c.classList.remove("is-on"); });
        chip.classList.add("is-on");
        var f = chip.getAttribute("data-filter");
        document.querySelectorAll(".forest tbody tr").forEach(function (row) {
          row.hidden = f !== "all" && row.getAttribute("data-dataset") !== f;
        });
      });
    });
  }

  function setupBeta() {
    var select = document.getElementById("beta-select");
    var table = document.getElementById("g2-table");
    if (!select || !table) return;
    select.addEventListener("change", function () {
      var key = select.value;
      table.querySelectorAll("tbody tr").forEach(function (row) {
        var total = row.getAttribute("data-total");
        var cert = row.getAttribute("data-c" + key);
        row.querySelector(".cert-c").textContent = cert + "/" + total;
        if (key === "020") {
          row.querySelector(".cert-f").textContent = row.getAttribute("data-f020");
          row.querySelector(".cert-u").textContent = row.getAttribute("data-u020");
        } else {
          row.querySelector(".cert-f").textContent = "—";
          row.querySelector(".cert-u").textContent = "—";
        }
      });
    });
  }

  function setupSort() {
    var btn = document.querySelector("#dior-table th[data-sort='n'] button");
    if (!btn) return;
    var desc = true;
    btn.addEventListener("click", function () {
      var tbody = document.querySelector("#dior-table tbody");
      var rows = Array.prototype.slice.call(tbody.querySelectorAll("tr"));
      rows.sort(function (a, b) {
        var na = Number(a.querySelector("[data-n]").getAttribute("data-n"));
        var nb = Number(b.querySelector("[data-n]").getAttribute("data-n"));
        return desc ? nb - na : na - nb;
      });
      desc = !desc;
      rows.forEach(function (r) { tbody.appendChild(r); });
    });
  }

  setupRotator();
  setupForest();
  setupBeta();
  setupSort();
})();
